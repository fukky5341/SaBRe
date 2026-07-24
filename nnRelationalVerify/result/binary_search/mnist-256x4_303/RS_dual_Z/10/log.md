## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 338.886556389
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-184.7211609, 146.8837128, -184.7211609, 146.8837128, -331.6048584, 331.6048584)
1: (-155.1049194, 130.5072479, -155.1049194, 130.5072479, -285.6120911, 285.6120911)
2: (-203.6430969, 131.9900055, -203.6430969, 131.9900055, -335.6330566, 335.6330566)
3: (-216.2613983, 114.2387161, -216.2613983, 114.2387161, -330.5000916, 330.5000916)
4: (-198.3916016, 151.6778870, -198.3916016, 151.6778870, -350.0694885, 350.0694885)
5: (-177.7157135, 138.1587830, -177.7157135, 138.1587830, -315.8745117, 315.8745117)
6: (-170.2996063, 163.9844208, -170.2996063, 163.9844208, -334.2840271, 334.2840271)
7: (-185.1827240, 156.0302124, -185.1827240, 156.0302124, -341.2129517, 341.2129517)
8: (-223.8251801, 153.0471344, -223.8251801, 153.0471344, -376.8723145, 376.8723145)
9: (-169.0843506, 166.1456757, -169.0843506, 166.1456757, -335.2300110, 335.2300110)

## BASE Result
execution time: IAR + LP analysis = 1.10 + 11.70 = 12.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -338.9345818, upper bound: 338.9345818


# Binary Search by BASE starts (time budget: 2687.21 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=341.21295166015625
rel_dist={7: [-338.93443485344073, 338.93443485344073]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=341.21295166015625
rel_dist={7: [-338.93400199054076, 338.93400199054076]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=341.21295166015625
rel_dist={7: [-338.93322554933513, 338.933225507835]}

## Binary Search Result
Binary search time: 44.98 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2642.23 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9251655, upper bound: 338.9251655
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9251655, upper bound: 338.9251655
time: 8.06 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.36
Output dim: 7, lower bound: -338.9251655, upper bound: 338.9251655
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.36
Output dim: 7, lower bound: -338.9251655, upper bound: 338.9251655

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -184.7211609, 146.8837128, -184.7211609, 146.8837128, -331.6048584, 331.6048584
1: -155.1049194, 130.5072479, -155.1049194, 130.5072479, -285.6120911, 285.6120911
2: -203.6430969, 131.9900055, -203.6430969, 131.9900055, -335.6330566, 335.6330566
3: -216.2613983, 114.2387161, -216.2613983, 114.2387161, -330.5000916, 330.5000916
4: -198.3916016, 151.6778870, -198.3916016, 151.6778870, -350.0694885, 350.0694885
5: -177.7157135, 138.1587830, -177.7157135, 138.1587830, -315.8745117, 315.8745117
6: -170.2996063, 163.9844208, -170.2996063, 163.9844208, -334.2840271, 334.2840271
7: -185.1827240, 156.0302124, -185.1827240, 156.0302124, -341.2129517, 341.2129517
8: -223.8251801, 153.0471344, -223.8251801, 153.0471344, -376.8723145, 376.8723145
9: -169.0843506, 166.1456757, -169.0843506, 166.1456757, -335.2300110, 335.2300110

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8733966, upper bound: 338.8733966
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8733966, upper bound: 338.8733966
time: 6.40 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -184.7211609, 146.8837128, -184.7211609, 146.8837128, -331.6048584, 331.6048584
1: -155.1049194, 130.5072479, -155.1049194, 130.5072479, -285.6120911, 285.6120911
2: -203.6430969, 131.9900055, -203.6430969, 131.9900055, -335.6330566, 335.6330566
3: -216.2613983, 114.2387161, -216.2613983, 114.2387161, -330.5000916, 330.5000916
4: -198.3916016, 151.6778870, -198.3916016, 151.6778870, -350.0694885, 350.0694885
5: -177.7157135, 138.1587830, -177.7157135, 138.1587830, -315.8745117, 315.8745117
6: -170.2996063, 163.9844208, -170.2996063, 163.9844208, -334.2840271, 334.2840271
7: -185.1827240, 156.0302124, -185.1827240, 156.0302124, -341.2129517, 341.2129517
8: -223.8251801, 153.0471344, -223.8251801, 153.0471344, -376.8723145, 376.8723145
9: -169.0843506, 166.1456757, -169.0843506, 166.1456757, -335.2300110, 335.2300110

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8733966, upper bound: 338.8733966
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8733966, upper bound: 338.8733966
time: 6.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.48 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 14.48
Output dim: 7, lower bound: -338.8733966, upper bound: 338.8733966
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 14.48
Output dim: 7, lower bound: -338.8733966, upper bound: 338.8733966
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 14.48
Output dim: 7, lower bound: -338.8733966, upper bound: 338.8733966
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 14.48
Output dim: 7, lower bound: -338.8733966, upper bound: 338.8733966
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=341.21295166015625
rel_dist={7: [-338.93443485344073, 338.93443485344073]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9252331, upper bound: 338.9252331
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9252331, upper bound: 338.9252331
time: 8.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.81
Output dim: 7, lower bound: -338.9252331, upper bound: 338.9252331
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.81
Output dim: 7, lower bound: -338.9252331, upper bound: 338.9252331

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -184.7211609, 146.8837128, -184.7211609, 146.8837128, -331.6048584, 331.6048584
1: -155.1049194, 130.5072479, -155.1049194, 130.5072479, -285.6120911, 285.6120911
2: -203.6430969, 131.9900055, -203.6430969, 131.9900055, -335.6330566, 335.6330566
3: -216.2613983, 114.2387161, -216.2613983, 114.2387161, -330.5000916, 330.5000916
4: -198.3916016, 151.6778870, -198.3916016, 151.6778870, -350.0694885, 350.0694885
5: -177.7157135, 138.1587830, -177.7157135, 138.1587830, -315.8745117, 315.8745117
6: -170.2996063, 163.9844208, -170.2996063, 163.9844208, -334.2840271, 334.2840271
7: -185.1827240, 156.0302124, -185.1827240, 156.0302124, -341.2129517, 341.2129517
8: -223.8251801, 153.0471344, -223.8251801, 153.0471344, -376.8723145, 376.8723145
9: -169.0843506, 166.1456757, -169.0843506, 166.1456757, -335.2300110, 335.2300110

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8734698, upper bound: 338.8734698
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8734698, upper bound: 338.8734698
time: 6.10 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -184.7211609, 146.8837128, -184.7211609, 146.8837128, -331.6048584, 331.6048584
1: -155.1049194, 130.5072479, -155.1049194, 130.5072479, -285.6120911, 285.6120911
2: -203.6430969, 131.9900055, -203.6430969, 131.9900055, -335.6330566, 335.6330566
3: -216.2613983, 114.2387161, -216.2613983, 114.2387161, -330.5000916, 330.5000916
4: -198.3916016, 151.6778870, -198.3916016, 151.6778870, -350.0694885, 350.0694885
5: -177.7157135, 138.1587830, -177.7157135, 138.1587830, -315.8745117, 315.8745117
6: -170.2996063, 163.9844208, -170.2996063, 163.9844208, -334.2840271, 334.2840271
7: -185.1827240, 156.0302124, -185.1827240, 156.0302124, -341.2129517, 341.2129517
8: -223.8251801, 153.0471344, -223.8251801, 153.0471344, -376.8723145, 376.8723145
9: -169.0843506, 166.1456757, -169.0843506, 166.1456757, -335.2300110, 335.2300110

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8734698, upper bound: 338.8734698
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8734698, upper bound: 338.8734698
time: 6.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.52 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 14.52
Output dim: 7, lower bound: -338.8734698, upper bound: 338.8734698
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 14.52
Output dim: 7, lower bound: -338.8734698, upper bound: 338.8734698
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 14.52
Output dim: 7, lower bound: -338.8734698, upper bound: 338.8734698
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 14.52
Output dim: 7, lower bound: -338.8734698, upper bound: 338.8734698
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=341.21295166015625
rel_dist={7: [-338.93451046142036, 338.9345104226113]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9252726, upper bound: 338.9252726
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9252726, upper bound: 338.9252726
time: 8.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.71 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.71
Output dim: 7, lower bound: -338.9252726, upper bound: 338.9252726
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.71
Output dim: 7, lower bound: -338.9252726, upper bound: 338.9252726

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -184.7211609, 146.8837128, -184.7211609, 146.8837128, -331.6048584, 331.6048584
1: -155.1049194, 130.5072479, -155.1049194, 130.5072479, -285.6120911, 285.6120911
2: -203.6430969, 131.9900055, -203.6430969, 131.9900055, -335.6330566, 335.6330566
3: -216.2613983, 114.2387161, -216.2613983, 114.2387161, -330.5000916, 330.5000916
4: -198.3916016, 151.6778870, -198.3916016, 151.6778870, -350.0694885, 350.0694885
5: -177.7157135, 138.1587830, -177.7157135, 138.1587830, -315.8745117, 315.8745117
6: -170.2996063, 163.9844208, -170.2996063, 163.9844208, -334.2840271, 334.2840271
7: -185.1827240, 156.0302124, -185.1827240, 156.0302124, -341.2129517, 341.2129517
8: -223.8251801, 153.0471344, -223.8251801, 153.0471344, -376.8723145, 376.8723145
9: -169.0843506, 166.1456757, -169.0843506, 166.1456757, -335.2300110, 335.2300110

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8735140, upper bound: 338.8735140
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8735140, upper bound: 338.8735140
time: 7.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -184.7211609, 146.8837128, -184.7211609, 146.8837128, -331.6048584, 331.6048584
1: -155.1049194, 130.5072479, -155.1049194, 130.5072479, -285.6120911, 285.6120911
2: -203.6430969, 131.9900055, -203.6430969, 131.9900055, -335.6330566, 335.6330566
3: -216.2613983, 114.2387161, -216.2613983, 114.2387161, -330.5000916, 330.5000916
4: -198.3916016, 151.6778870, -198.3916016, 151.6778870, -350.0694885, 350.0694885
5: -177.7157135, 138.1587830, -177.7157135, 138.1587830, -315.8745117, 315.8745117
6: -170.2996063, 163.9844208, -170.2996063, 163.9844208, -334.2840271, 334.2840271
7: -185.1827240, 156.0302124, -185.1827240, 156.0302124, -341.2129517, 341.2129517
8: -223.8251801, 153.0471344, -223.8251801, 153.0471344, -376.8723145, 376.8723145
9: -169.0843506, 166.1456757, -169.0843506, 166.1456757, -335.2300110, 335.2300110

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8735140, upper bound: 338.8735140
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8735140, upper bound: 338.8735140
time: 6.96 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.01 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 15.01
Output dim: 7, lower bound: -338.8735140, upper bound: 338.8735140
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 15.01
Output dim: 7, lower bound: -338.8735140, upper bound: 338.8735140
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 15.01
Output dim: 7, lower bound: -338.8735140, upper bound: 338.8735140
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 15.01
Output dim: 7, lower bound: -338.8735140, upper bound: 338.8735140
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=341.21295166015625
rel_dist={7: [-338.9345586391237, 338.9345586391237]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9252919, upper bound: 338.9252919
time: 7.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9252919, upper bound: 338.9252919
time: 9.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.00 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.00
Output dim: 7, lower bound: -338.9252919, upper bound: 338.9252919
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.00
Output dim: 7, lower bound: -338.9252919, upper bound: 338.9252919

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -184.7211609, 146.8837128, -184.7211609, 146.8837128, -331.6048584, 331.6048584
1: -155.1049194, 130.5072479, -155.1049194, 130.5072479, -285.6120911, 285.6120911
2: -203.6430969, 131.9900055, -203.6430969, 131.9900055, -335.6330566, 335.6330566
3: -216.2613983, 114.2387161, -216.2613983, 114.2387161, -330.5000916, 330.5000916
4: -198.3916016, 151.6778870, -198.3916016, 151.6778870, -350.0694885, 350.0694885
5: -177.7157135, 138.1587830, -177.7157135, 138.1587830, -315.8745117, 315.8745117
6: -170.2996063, 163.9844208, -170.2996063, 163.9844208, -334.2840271, 334.2840271
7: -185.1827240, 156.0302124, -185.1827240, 156.0302124, -341.2129517, 341.2129517
8: -223.8251801, 153.0471344, -223.8251801, 153.0471344, -376.8723145, 376.8723145
9: -169.0843506, 166.1456757, -169.0843506, 166.1456757, -335.2300110, 335.2300110

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8735350, upper bound: 338.8735350
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8735350, upper bound: 338.8735350
time: 6.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -184.7211609, 146.8837128, -184.7211609, 146.8837128, -331.6048584, 331.6048584
1: -155.1049194, 130.5072479, -155.1049194, 130.5072479, -285.6120911, 285.6120911
2: -203.6430969, 131.9900055, -203.6430969, 131.9900055, -335.6330566, 335.6330566
3: -216.2613983, 114.2387161, -216.2613983, 114.2387161, -330.5000916, 330.5000916
4: -198.3916016, 151.6778870, -198.3916016, 151.6778870, -350.0694885, 350.0694885
5: -177.7157135, 138.1587830, -177.7157135, 138.1587830, -315.8745117, 315.8745117
6: -170.2996063, 163.9844208, -170.2996063, 163.9844208, -334.2840271, 334.2840271
7: -185.1827240, 156.0302124, -185.1827240, 156.0302124, -341.2129517, 341.2129517
8: -223.8251801, 153.0471344, -223.8251801, 153.0471344, -376.8723145, 376.8723145
9: -169.0843506, 166.1456757, -169.0843506, 166.1456757, -335.2300110, 335.2300110

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8735350, upper bound: 338.8735350
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8735350, upper bound: 338.8735350
time: 6.11 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.52 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 13.52
Output dim: 7, lower bound: -338.8735350, upper bound: 338.8735350
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 13.52
Output dim: 7, lower bound: -338.8735350, upper bound: 338.8735350
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 13.52
Output dim: 7, lower bound: -338.8735350, upper bound: 338.8735350
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 13.52
Output dim: 7, lower bound: -338.8735350, upper bound: 338.8735350
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=341.21295166015625
rel_dist={7: [-338.93458180562635, 338.93458176585284]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 236.10 seconds
