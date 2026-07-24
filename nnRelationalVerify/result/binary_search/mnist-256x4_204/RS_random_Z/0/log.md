## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 587.735384174
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733)
1: (-275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269)
2: (-361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050)
3: (-382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456)
4: (-352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934)
5: (-314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059)
6: (-301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518)
7: (-328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479)
8: (-396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159)
9: (-298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676)

## BASE Result
execution time: IAR + LP analysis = 1.06 + 12.85 = 13.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -587.7908408, upper bound: 587.7908408


# Binary Search by BASE starts (time budget: 2686.10 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=591.3155517578125
rel_dist={6: [-587.7907620297522, 587.7907620249766]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=591.3155517578125
rel_dist={6: [-587.7906229930563, 587.7906229976039]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=591.3155517578125
rel_dist={6: [-587.7904223681265, 587.7904223711924]}

## Binary Search Result
Binary search time: 50.64 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2635.45 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7893397, upper bound: 587.7893514
time: 10.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7893514, upper bound: 587.7893397
time: 9.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 20.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 20.04
Output dim: 6, lower bound: -587.7893397, upper bound: 587.7893514
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 20.04
Output dim: 6, lower bound: -587.7893514, upper bound: 587.7893397

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7345969, upper bound: 587.7346235
time: 9.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7345969, upper bound: 587.7346235
time: 9.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7855044, upper bound: 587.7855044
time: 10.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7855044, upper bound: 587.7855044
time: 10.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.72 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 21.72
Output dim: 6, lower bound: -587.7345969, upper bound: 587.7346235
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 21.72
Output dim: 6, lower bound: -587.7345969, upper bound: 587.7346235
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.72
Output dim: 6, lower bound: -587.7855044, upper bound: 587.7855044
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.72
Output dim: 6, lower bound: -587.7855044, upper bound: 587.7855044

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7705273, upper bound: 587.7705155
time: 9.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7705273, upper bound: 587.7705155
time: 10.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854913, upper bound: 587.7855044
time: 10.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7855044, upper bound: 587.7854926
time: 10.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.61 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.61
Output dim: 6, lower bound: -587.7705273, upper bound: 587.7705155
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.61
Output dim: 6, lower bound: -587.7705273, upper bound: 587.7705155
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.61
Output dim: 6, lower bound: -587.7854913, upper bound: 587.7855044
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.61
Output dim: 6, lower bound: -587.7855044, upper bound: 587.7854926

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7621723, upper bound: 587.7621709
time: 10.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7621723, upper bound: 587.7621709
time: 10.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7695320, upper bound: 587.7694801
time: 9.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7694901, upper bound: 587.7695217
time: 9.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 193

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7534728, upper bound: 587.7534788
time: 9.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7534728, upper bound: 587.7534788
time: 9.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7670134, upper bound: 587.7669871
time: 9.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7670134, upper bound: 587.7669871
time: 9.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.17 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.17
Output dim: 6, lower bound: -587.7621723, upper bound: 587.7621709
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.17
Output dim: 6, lower bound: -587.7621723, upper bound: 587.7621709
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.17
Output dim: 6, lower bound: -587.7695320, upper bound: 587.7694801
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.17
Output dim: 6, lower bound: -587.7694901, upper bound: 587.7695217
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.17
Output dim: 6, lower bound: -587.7534728, upper bound: 587.7534788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.17
Output dim: 6, lower bound: -587.7534728, upper bound: 587.7534788
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.17
Output dim: 6, lower bound: -587.7670134, upper bound: 587.7669871
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.17
Output dim: 6, lower bound: -587.7670134, upper bound: 587.7669871

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7621723, upper bound: 587.7621685
time: 9.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7621702, upper bound: 587.7621709
time: 9.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7464519, upper bound: 587.7464417
time: 9.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7464519, upper bound: 587.7464417
time: 9.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7517364, upper bound: 587.7516872
time: 8.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7517364, upper bound: 587.7516872
time: 8.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7628671, upper bound: 587.7628676
time: 9.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7628671, upper bound: 587.7628676
time: 9.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7534696, upper bound: 587.7534788
time: 9.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7534728, upper bound: 587.7534762
time: 8.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7429311, upper bound: 587.7429078
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7429032, upper bound: 587.7429395
time: 9.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7477309, upper bound: 587.7476960
time: 8.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7477309, upper bound: 587.7476960
time: 8.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7604264, upper bound: 587.7604098
time: 9.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7604264, upper bound: 587.7604098
time: 10.20 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.74 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7621723, upper bound: 587.7621685
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7621702, upper bound: 587.7621709
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7464519, upper bound: 587.7464417
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7464519, upper bound: 587.7464417
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7517364, upper bound: 587.7516872
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7517364, upper bound: 587.7516872
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7628671, upper bound: 587.7628676
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7628671, upper bound: 587.7628676
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7534696, upper bound: 587.7534788
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7534728, upper bound: 587.7534762
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7429311, upper bound: 587.7429078
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7429032, upper bound: 587.7429395
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7477309, upper bound: 587.7476960
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7477309, upper bound: 587.7476960
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7604264, upper bound: 587.7604098
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.74
Output dim: 6, lower bound: -587.7604264, upper bound: 587.7604098

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7498620, upper bound: 587.7498617
time: 10.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7498620, upper bound: 587.7498617
time: 11.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7537498, upper bound: 587.7537499
time: 8.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7537498, upper bound: 587.7537499
time: 9.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7464519, upper bound: 587.7464417
time: 10.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7464519, upper bound: 587.7464417
time: 9.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7384370, upper bound: 587.7384452
time: 10.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7384370, upper bound: 587.7384452
time: 10.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7507068, upper bound: 587.7506645
time: 8.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7507059, upper bound: 587.7506508
time: 11.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7197625, upper bound: 587.7197563
time: 9.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7197625, upper bound: 587.7197563
time: 9.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 139

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7088763, upper bound: 587.7089227
time: 8.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7088763, upper bound: 587.7089227
time: 8.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7628470, upper bound: 587.7628676
time: 10.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7628671, upper bound: 587.7628533
time: 11.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7434221, upper bound: 587.7434411
time: 9.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7434214, upper bound: 587.7434412
time: 9.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7474451, upper bound: 587.7474307
time: 9.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7474451, upper bound: 587.7474307
time: 9.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7068152, upper bound: 587.7068184
time: 10.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7068152, upper bound: 587.7068184
time: 10.08 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 25.31 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7498620, upper bound: 587.7498617
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7498620, upper bound: 587.7498617
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7537498, upper bound: 587.7537499
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7537498, upper bound: 587.7537499
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7464519, upper bound: 587.7464417
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7464519, upper bound: 587.7464417
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7384370, upper bound: 587.7384452
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7384370, upper bound: 587.7384452
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7507068, upper bound: 587.7506645
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7507059, upper bound: 587.7506508
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7197625, upper bound: 587.7197563
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7197625, upper bound: 587.7197563
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7088763, upper bound: 587.7089227
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7088763, upper bound: 587.7089227
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7628470, upper bound: 587.7628676
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7628671, upper bound: 587.7628533
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7434221, upper bound: 587.7434411
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7434214, upper bound: 587.7434412
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7474451, upper bound: 587.7474307
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7474451, upper bound: 587.7474307
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7068152, upper bound: 587.7068184
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.31
Output dim: 6, lower bound: -587.7068152, upper bound: 587.7068184
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.31
Output dim: 6, lower bound: -587.7429032, upper bound: 587.7429395
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.31
Output dim: 6, lower bound: -587.7477309, upper bound: 587.7476960
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.31
Output dim: 6, lower bound: -587.7477309, upper bound: 587.7476960
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.31
Output dim: 6, lower bound: -587.7604264, upper bound: 587.7604098
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.31
Output dim: 6, lower bound: -587.7604264, upper bound: 587.7604098
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=591.3155517578125
rel_dist={6: [-587.7907620297522, 587.7907620249766]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7889653, upper bound: 587.7889653
time: 11.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7889653, upper bound: 587.7889653
time: 12.12 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 23.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 23.90
Output dim: 6, lower bound: -587.7889653, upper bound: 587.7889653
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 23.90
Output dim: 6, lower bound: -587.7889653, upper bound: 587.7889653

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7076247, upper bound: 587.7076247
time: 8.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7076247, upper bound: 587.7076247
time: 8.09 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7850051, upper bound: 587.7850051
time: 11.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7850051, upper bound: 587.7850051
time: 12.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.42 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 24.42
Output dim: 6, lower bound: -587.7076247, upper bound: 587.7076247
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 24.42
Output dim: 6, lower bound: -587.7076247, upper bound: 587.7076247
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.42
Output dim: 6, lower bound: -587.7850051, upper bound: 587.7850051
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.42
Output dim: 6, lower bound: -587.7850051, upper bound: 587.7850051

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7850034, upper bound: 587.7850051
time: 11.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7850051, upper bound: 587.7850034
time: 12.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7849897, upper bound: 587.7849926
time: 12.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7849927, upper bound: 587.7849897
time: 11.51 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.49 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.49
Output dim: 6, lower bound: -587.7850034, upper bound: 587.7850051
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.49
Output dim: 6, lower bound: -587.7850051, upper bound: 587.7850034
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.49
Output dim: 6, lower bound: -587.7849897, upper bound: 587.7849926
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.49
Output dim: 6, lower bound: -587.7849927, upper bound: 587.7849897

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7844215, upper bound: 587.7844250
time: 12.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7844215, upper bound: 587.7844250
time: 11.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7609853, upper bound: 587.7609812
time: 11.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7609853, upper bound: 587.7609812
time: 11.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7849884, upper bound: 587.7849913
time: 13.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7849884, upper bound: 587.7849913
time: 10.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7839086, upper bound: 587.7839070
time: 12.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7839086, upper bound: 587.7839070
time: 11.53 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.42 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 6, lower bound: -587.7844215, upper bound: 587.7844250
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 6, lower bound: -587.7844215, upper bound: 587.7844250
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 6, lower bound: -587.7609853, upper bound: 587.7609812
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 6, lower bound: -587.7609853, upper bound: 587.7609812
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 6, lower bound: -587.7849884, upper bound: 587.7849913
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 6, lower bound: -587.7849884, upper bound: 587.7849913
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 6, lower bound: -587.7839086, upper bound: 587.7839070
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 6, lower bound: -587.7839086, upper bound: 587.7839070

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7821943, upper bound: 587.7821949
time: 12.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7821943, upper bound: 587.7821949
time: 12.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7666627, upper bound: 587.7666628
time: 10.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7666627, upper bound: 587.7666628
time: 10.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7538250, upper bound: 587.7538249
time: 9.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7538250, upper bound: 587.7538249
time: 10.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7608169, upper bound: 587.7608030
time: 10.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7608069, upper bound: 587.7608139
time: 11.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 130

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7843604, upper bound: 587.7843660
time: 11.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7843604, upper bound: 587.7843659
time: 11.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7819559, upper bound: 587.7819639
time: 11.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7819559, upper bound: 587.7819639
time: 15.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7658585, upper bound: 587.7658497
time: 10.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7658585, upper bound: 587.7658497
time: 10.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7779368, upper bound: 587.7779306
time: 11.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7779368, upper bound: 587.7779306
time: 11.46 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.06 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7821943, upper bound: 587.7821949
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7821943, upper bound: 587.7821949
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7666627, upper bound: 587.7666628
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7666627, upper bound: 587.7666628
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7538250, upper bound: 587.7538249
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7538250, upper bound: 587.7538249
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7608169, upper bound: 587.7608030
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7608069, upper bound: 587.7608139
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7843604, upper bound: 587.7843660
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7843604, upper bound: 587.7843659
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7819559, upper bound: 587.7819639
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7819559, upper bound: 587.7819639
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7658585, upper bound: 587.7658497
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7658585, upper bound: 587.7658497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7779368, upper bound: 587.7779306
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.06
Output dim: 6, lower bound: -587.7779368, upper bound: 587.7779306

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7820608, upper bound: 587.7820810
time: 12.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7820801, upper bound: 587.7820637
time: 11.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7759946, upper bound: 587.7759958
time: 11.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7759946, upper bound: 587.7759958
time: 11.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7666592, upper bound: 587.7666628
time: 10.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7666627, upper bound: 587.7666591
time: 11.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7637098, upper bound: 587.7637152
time: 12.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7637098, upper bound: 587.7637152
time: 10.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7480074, upper bound: 587.7480190
time: 11.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7480169, upper bound: 587.7480094
time: 11.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7283286, upper bound: 587.7283289
time: 9.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7283286, upper bound: 587.7283289
time: 9.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7396159, upper bound: 587.7396089
time: 11.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7396159, upper bound: 587.7396089
time: 11.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=591.3155517578125
rel_dist={6: [-587.7906229930563, 587.7906229976039]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7869006, upper bound: 587.7869006
time: 17.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7869006, upper bound: 587.7869007
time: 18.12 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 35.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 35.65
Output dim: 6, lower bound: -587.7869006, upper bound: 587.7869006
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 35.65
Output dim: 6, lower bound: -587.7869006, upper bound: 587.7869007

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7366621, upper bound: 587.7366588
time: 12.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7366621, upper bound: 587.7366588
time: 12.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7868913, upper bound: 587.7868962
time: 18.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7868962, upper bound: 587.7868915
time: 16.07 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 35.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 35.57
Output dim: 6, lower bound: -587.7366621, upper bound: 587.7366588
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 35.57
Output dim: 6, lower bound: -587.7366621, upper bound: 587.7366588
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 35.57
Output dim: 6, lower bound: -587.7868913, upper bound: 587.7868962
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 35.57
Output dim: 6, lower bound: -587.7868962, upper bound: 587.7868915

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7182733, upper bound: 587.7182733
time: 12.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7182733, upper bound: 587.7182733
time: 12.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7305589, upper bound: 587.7305588
time: 12.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7305589, upper bound: 587.7305588
time: 13.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7850521, upper bound: 587.7850580
time: 19.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7850521, upper bound: 587.7850580
time: 17.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7791672, upper bound: 587.7791602
time: 16.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7791672, upper bound: 587.7791601
time: 17.95 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 34.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 34.94
Output dim: 6, lower bound: -587.7182733, upper bound: 587.7182733
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 34.94
Output dim: 6, lower bound: -587.7182733, upper bound: 587.7182733
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 34.94
Output dim: 6, lower bound: -587.7305589, upper bound: 587.7305588
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 34.94
Output dim: 6, lower bound: -587.7305589, upper bound: 587.7305588
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 34.94
Output dim: 6, lower bound: -587.7850521, upper bound: 587.7850580
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 34.94
Output dim: 6, lower bound: -587.7850521, upper bound: 587.7850580
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 34.94
Output dim: 6, lower bound: -587.7791672, upper bound: 587.7791602
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 34.94
Output dim: 6, lower bound: -587.7791672, upper bound: 587.7791601

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7850502, upper bound: 587.7850580
time: 16.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7850502, upper bound: 587.7850548
time: 16.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7850519, upper bound: 587.7850579
time: 15.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7850519, upper bound: 587.7850580
time: 17.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7791670, upper bound: 587.7791602
time: 16.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7791672, upper bound: 587.7791601
time: 14.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7780169, upper bound: 587.7780098
time: 21.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7780165, upper bound: 587.7780110
time: 17.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 39.96 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 39.96
Output dim: 6, lower bound: -587.7850502, upper bound: 587.7850580
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 39.96
Output dim: 6, lower bound: -587.7850502, upper bound: 587.7850548
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 39.96
Output dim: 6, lower bound: -587.7850519, upper bound: 587.7850579
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 39.96
Output dim: 6, lower bound: -587.7850519, upper bound: 587.7850580
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 39.96
Output dim: 6, lower bound: -587.7791670, upper bound: 587.7791602
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 39.96
Output dim: 6, lower bound: -587.7791672, upper bound: 587.7791601
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 39.96
Output dim: 6, lower bound: -587.7780169, upper bound: 587.7780098
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 39.96
Output dim: 6, lower bound: -587.7780165, upper bound: 587.7780110

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7816307, upper bound: 587.7816402
time: 17.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7816307, upper bound: 587.7816402
time: 22.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7724390, upper bound: 587.7724416
time: 18.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7724390, upper bound: 587.7724416
time: 19.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7844043, upper bound: 587.7844079
time: 18.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7844037, upper bound: 587.7844079
time: 18.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7784507, upper bound: 587.7784510
time: 17.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7784507, upper bound: 587.7784510
time: 15.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7338099, upper bound: 587.7338122
time: 13.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7338099, upper bound: 587.7338122
time: 13.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733
1: -275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269
2: -361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050
3: -382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456
4: -352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934
5: -314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059
6: -301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518
7: -328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479
8: -396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159
9: -298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7746522, upper bound: 587.7746535
time: 16.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7746522, upper bound: 587.7746535
time: 16.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 33.75 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7816307, upper bound: 587.7816402
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7816307, upper bound: 587.7816402
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7724390, upper bound: 587.7724416
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7724390, upper bound: 587.7724416
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7844043, upper bound: 587.7844079
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7844037, upper bound: 587.7844079
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7784507, upper bound: 587.7784510
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7784507, upper bound: 587.7784510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7338099, upper bound: 587.7338122
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7338099, upper bound: 587.7338122
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7746522, upper bound: 587.7746535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 33.75
Output dim: 6, lower bound: -587.7746522, upper bound: 587.7746535
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 6, lower bound: -587.7780169, upper bound: 587.7780098
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 6, lower bound: -587.7780165, upper bound: 587.7780110
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=591.3155517578125
rel_dist={6: [-587.7904223681265, 587.7904223711924]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1822.65 seconds
