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
execution time: IAR + LP analysis = 1.05 + 12.55 = 13.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -587.7908408, upper bound: 587.7908408


# Binary Search by BASE starts (time budget: 2686.40 seconds, max iter: 100)

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
Binary search time: 50.68 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2635.72 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7907521, upper bound: 587.7907620
time: 9.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7907620, upper bound: 587.7907521
time: 9.95 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.83
Output dim: 6, lower bound: -587.7907521, upper bound: 587.7907620
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.83
Output dim: 6, lower bound: -587.7907620, upper bound: 587.7907521

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7560854, upper bound: 587.7560896
time: 8.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7560854, upper bound: 587.7560896
time: 8.66 seconds

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7560896, upper bound: 587.7560854
time: 9.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7560896, upper bound: 587.7560854
time: 9.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.86
Output dim: 6, lower bound: -587.7560854, upper bound: 587.7560896
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.86
Output dim: 6, lower bound: -587.7560854, upper bound: 587.7560896
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.86
Output dim: 6, lower bound: -587.7560896, upper bound: 587.7560854
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.86
Output dim: 6, lower bound: -587.7560896, upper bound: 587.7560854

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7153977, upper bound: 587.7154118
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7153977, upper bound: 587.7154118
time: 6.85 seconds

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7153977, upper bound: 587.7154118
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7153977, upper bound: 587.7154118
time: 6.85 seconds

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154118, upper bound: 587.7153977
time: 8.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154118, upper bound: 587.7153977
time: 8.07 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154118, upper bound: 587.7153977
time: 8.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154118, upper bound: 587.7153977
time: 9.21 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.49 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.49
Output dim: 6, lower bound: -587.7153977, upper bound: 587.7154118
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.49
Output dim: 6, lower bound: -587.7153977, upper bound: 587.7154118
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.49
Output dim: 6, lower bound: -587.7153977, upper bound: 587.7154118
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.49
Output dim: 6, lower bound: -587.7153977, upper bound: 587.7154118
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.49
Output dim: 6, lower bound: -587.7154118, upper bound: 587.7153977
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.49
Output dim: 6, lower bound: -587.7154118, upper bound: 587.7153977
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.49
Output dim: 6, lower bound: -587.7154118, upper bound: 587.7153977
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.49
Output dim: 6, lower bound: -587.7154118, upper bound: 587.7153977
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=591.3155517578125
rel_dist={6: [-587.7907620297522, 587.7907620249766]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7907936, upper bound: 587.7908094
time: 8.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7908094, upper bound: 587.7907936
time: 9.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.19 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.19
Output dim: 6, lower bound: -587.7907936, upper bound: 587.7908094
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.19
Output dim: 6, lower bound: -587.7908094, upper bound: 587.7907936

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

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561243, upper bound: 587.7561292
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561243, upper bound: 587.7561292
time: 8.45 seconds

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561292, upper bound: 587.7561243
time: 9.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561292, upper bound: 587.7561243
time: 9.45 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.99
Output dim: 6, lower bound: -587.7561243, upper bound: 587.7561292
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.99
Output dim: 6, lower bound: -587.7561243, upper bound: 587.7561292
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.99
Output dim: 6, lower bound: -587.7561292, upper bound: 587.7561243
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.99
Output dim: 6, lower bound: -587.7561292, upper bound: 587.7561243

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154318, upper bound: 587.7154460
time: 9.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154318, upper bound: 587.7154460
time: 9.24 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154318, upper bound: 587.7154460
time: 9.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154318, upper bound: 587.7154460
time: 9.17 seconds

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154460, upper bound: 587.7154318
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154460, upper bound: 587.7154318
time: 7.57 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154460, upper bound: 587.7154318
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154460, upper bound: 587.7154318
time: 7.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 16.27
Output dim: 6, lower bound: -587.7154318, upper bound: 587.7154460
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 16.27
Output dim: 6, lower bound: -587.7154318, upper bound: 587.7154460
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 16.27
Output dim: 6, lower bound: -587.7154318, upper bound: 587.7154460
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 16.27
Output dim: 6, lower bound: -587.7154318, upper bound: 587.7154460
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 16.27
Output dim: 6, lower bound: -587.7154460, upper bound: 587.7154318
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 16.27
Output dim: 6, lower bound: -587.7154460, upper bound: 587.7154318
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 16.27
Output dim: 6, lower bound: -587.7154460, upper bound: 587.7154318
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 16.27
Output dim: 6, lower bound: -587.7154460, upper bound: 587.7154318
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=591.3155517578125
rel_dist={6: [-587.7908094388836, 587.7908094449704]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7908109, upper bound: 587.7908306
time: 9.37 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7908306, upper bound: 587.7908109
time: 7.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.32
Output dim: 6, lower bound: -587.7908109, upper bound: 587.7908306
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.32
Output dim: 6, lower bound: -587.7908306, upper bound: 587.7908109

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561470, upper bound: 587.7561545
time: 9.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561470, upper bound: 587.7561545
time: 9.82 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561545, upper bound: 587.7561470
time: 8.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561545, upper bound: 587.7561470
time: 8.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.09
Output dim: 6, lower bound: -587.7561470, upper bound: 587.7561545
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.09
Output dim: 6, lower bound: -587.7561470, upper bound: 587.7561545
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.09
Output dim: 6, lower bound: -587.7561545, upper bound: 587.7561470
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.09
Output dim: 6, lower bound: -587.7561545, upper bound: 587.7561470

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154516, upper bound: 587.7154645
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154516, upper bound: 587.7154645
time: 8.49 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154516, upper bound: 587.7154645
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154516, upper bound: 587.7154645
time: 8.45 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154645, upper bound: 587.7154516
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154645, upper bound: 587.7154516
time: 8.07 seconds

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154645, upper bound: 587.7154516
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154645, upper bound: 587.7154516
time: 8.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 17.25
Output dim: 6, lower bound: -587.7154516, upper bound: 587.7154645
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 17.25
Output dim: 6, lower bound: -587.7154516, upper bound: 587.7154645
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 17.25
Output dim: 6, lower bound: -587.7154516, upper bound: 587.7154645
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 17.25
Output dim: 6, lower bound: -587.7154516, upper bound: 587.7154645
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 17.25
Output dim: 6, lower bound: -587.7154645, upper bound: 587.7154516
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 17.25
Output dim: 6, lower bound: -587.7154645, upper bound: 587.7154516
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 17.25
Output dim: 6, lower bound: -587.7154645, upper bound: 587.7154516
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 17.25
Output dim: 6, lower bound: -587.7154645, upper bound: 587.7154516
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=591.3155517578125
rel_dist={6: [-587.7908306045103, 587.7908306068607]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7908193, upper bound: 587.7908408
time: 8.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7908408, upper bound: 587.7908193
time: 8.86 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.78
Output dim: 6, lower bound: -587.7908193, upper bound: 587.7908408
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.78
Output dim: 6, lower bound: -587.7908408, upper bound: 587.7908193

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561579, upper bound: 587.7561670
time: 9.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561579, upper bound: 587.7561670
time: 9.48 seconds

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

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561670, upper bound: 587.7561579
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7561670, upper bound: 587.7561579
time: 7.91 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.64 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.64
Output dim: 6, lower bound: -587.7561579, upper bound: 587.7561670
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.64
Output dim: 6, lower bound: -587.7561579, upper bound: 587.7561670
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.64
Output dim: 6, lower bound: -587.7561670, upper bound: 587.7561579
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.64
Output dim: 6, lower bound: -587.7561670, upper bound: 587.7561579

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154615, upper bound: 587.7154727
time: 8.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154615, upper bound: 587.7154727
time: 8.15 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154615, upper bound: 587.7154727
time: 8.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154615, upper bound: 587.7154727
time: 8.13 seconds

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154727, upper bound: 587.7154615
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154727, upper bound: 587.7154615
time: 7.02 seconds

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

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154727, upper bound: 587.7154615
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7154727, upper bound: 587.7154615
time: 7.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 14.85
Output dim: 6, lower bound: -587.7154615, upper bound: 587.7154727
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 14.85
Output dim: 6, lower bound: -587.7154615, upper bound: 587.7154727
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 14.85
Output dim: 6, lower bound: -587.7154615, upper bound: 587.7154727
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 14.85
Output dim: 6, lower bound: -587.7154615, upper bound: 587.7154727
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 14.85
Output dim: 6, lower bound: -587.7154727, upper bound: 587.7154615
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 14.85
Output dim: 6, lower bound: -587.7154727, upper bound: 587.7154615
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 14.85
Output dim: 6, lower bound: -587.7154727, upper bound: 587.7154615
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 14.85
Output dim: 6, lower bound: -587.7154727, upper bound: 587.7154615
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=591.3155517578125
rel_dist={6: [-587.790840790525, 587.790840790525]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 552.48 seconds
