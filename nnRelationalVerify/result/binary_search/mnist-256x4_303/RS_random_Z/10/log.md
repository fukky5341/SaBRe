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
execution time: IAR + LP analysis = 1.11 + 11.68 = 12.79 seconds
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
Binary search time: 44.79 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2642.42 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9324499, upper bound: 338.9324499
time: 9.93 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9324499, upper bound: 338.9324499
time: 8.21 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.15
Output dim: 7, lower bound: -338.9324499, upper bound: 338.9324499
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.15
Output dim: 7, lower bound: -338.9324499, upper bound: 338.9324499

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9316834, upper bound: 338.9316834
time: 10.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9316834, upper bound: 338.9316834
time: 9.95 seconds

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9311433, upper bound: 338.9311453
time: 8.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9311453, upper bound: 338.9311433
time: 8.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.36 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.36
Output dim: 7, lower bound: -338.9316834, upper bound: 338.9316834
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.36
Output dim: 7, lower bound: -338.9316834, upper bound: 338.9316834
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.36
Output dim: 7, lower bound: -338.9311433, upper bound: 338.9311453
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.36
Output dim: 7, lower bound: -338.9311453, upper bound: 338.9311433

## BFS RS instance: RS_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8924043, upper bound: 338.8924043
time: 8.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8924043, upper bound: 338.8924043
time: 8.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9238595, upper bound: 338.9238595
time: 11.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9238595, upper bound: 338.9238595
time: 10.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9301899, upper bound: 338.9301942
time: 9.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9301908, upper bound: 338.9301937
time: 9.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9311445, upper bound: 338.9311433
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9311453, upper bound: 338.9311432
time: 9.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.66 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -338.8924043, upper bound: 338.8924043
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -338.8924043, upper bound: 338.8924043
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -338.9238595, upper bound: 338.9238595
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -338.9238595, upper bound: 338.9238595
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -338.9301899, upper bound: 338.9301942
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -338.9301908, upper bound: 338.9301937
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -338.9311445, upper bound: 338.9311433
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.66
Output dim: 7, lower bound: -338.9311453, upper bound: 338.9311432

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874428, upper bound: 338.8874428
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874428, upper bound: 338.8874428
time: 7.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8727945, upper bound: 338.8727945
time: 7.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8727945, upper bound: 338.8727945
time: 6.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9238123, upper bound: 338.9238123
time: 9.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9238123, upper bound: 338.9238123
time: 9.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8891642, upper bound: 338.8891642
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8891642, upper bound: 338.8891642
time: 8.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9299960, upper bound: 338.9299963
time: 10.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9299944, upper bound: 338.9299973
time: 9.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 162

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9124563, upper bound: 338.9124809
time: 9.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9124563, upper bound: 338.9124809
time: 9.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9311436, upper bound: 338.9311433
time: 8.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9311445, upper bound: 338.9311429
time: 9.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9311401, upper bound: 338.9311432
time: 9.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9311453, upper bound: 338.9311348
time: 8.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.8874428, upper bound: 338.8874428
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.8874428, upper bound: 338.8874428
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.8727945, upper bound: 338.8727945
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.8727945, upper bound: 338.8727945
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.9238123, upper bound: 338.9238123
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.9238123, upper bound: 338.9238123
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.8891642, upper bound: 338.8891642
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.8891642, upper bound: 338.8891642
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.9299960, upper bound: 338.9299963
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.9299944, upper bound: 338.9299973
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.9124563, upper bound: 338.9124809
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.9124563, upper bound: 338.9124809
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.9311436, upper bound: 338.9311433
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.9311445, upper bound: 338.9311429
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.9311401, upper bound: 338.9311432
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 7, lower bound: -338.9311453, upper bound: 338.9311348

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874428, upper bound: 338.8874380
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874380, upper bound: 338.8874428
time: 7.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8870570, upper bound: 338.8869671
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8869671, upper bound: 338.8870570
time: 8.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8809229, upper bound: 338.8809229
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8809229, upper bound: 338.8809229
time: 7.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9233753, upper bound: 338.9233777
time: 9.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9233777, upper bound: 338.9233753
time: 10.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8820752, upper bound: 338.8820299
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8820299, upper bound: 338.8820752
time: 9.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8889403, upper bound: 338.8888998
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8888998, upper bound: 338.8889403
time: 7.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9217635, upper bound: 338.9217538
time: 8.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9217635, upper bound: 338.9217538
time: 8.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9275536, upper bound: 338.9275496
time: 9.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9275536, upper bound: 338.9275496
time: 9.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9020448, upper bound: 338.9021090
time: 8.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9020448, upper bound: 338.9021090
time: 9.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8922880, upper bound: 338.8923167
time: 9.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8922880, upper bound: 338.8923167
time: 9.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026839, upper bound: 338.9026746
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026839, upper bound: 338.9026746
time: 8.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9310869, upper bound: 338.9310530
time: 9.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9310551, upper bound: 338.9310845
time: 10.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9262358, upper bound: 338.9261967
time: 9.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9262358, upper bound: 338.9261967
time: 9.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8960066, upper bound: 338.8960293
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8960066, upper bound: 338.8960293
time: 8.09 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 17.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8874428, upper bound: 338.8874380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8874380, upper bound: 338.8874428
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8870570, upper bound: 338.8869671
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8869671, upper bound: 338.8870570
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8809229, upper bound: 338.8809229
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8809229, upper bound: 338.8809229
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9233753, upper bound: 338.9233777
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9233777, upper bound: 338.9233753
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8820752, upper bound: 338.8820299
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8820299, upper bound: 338.8820752
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8889403, upper bound: 338.8888998
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8888998, upper bound: 338.8889403
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9217635, upper bound: 338.9217538
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9217635, upper bound: 338.9217538
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9275536, upper bound: 338.9275496
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9275536, upper bound: 338.9275496
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9020448, upper bound: 338.9021090
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9020448, upper bound: 338.9021090
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8922880, upper bound: 338.8923167
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8922880, upper bound: 338.8923167
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9026839, upper bound: 338.9026746
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9026839, upper bound: 338.9026746
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9310869, upper bound: 338.9310530
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9310551, upper bound: 338.9310845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9262358, upper bound: 338.9261967
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.9262358, upper bound: 338.9261967
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8960066, upper bound: 338.8960293
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 7, lower bound: -338.8960066, upper bound: 338.8960293

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8809200, upper bound: 338.8809189
time: 8.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8809200, upper bound: 338.8809189
time: 8.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8872935, upper bound: 338.8872978
time: 8.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8872936, upper bound: 338.8872985
time: 7.66 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 21.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 21.63
Output dim: 7, lower bound: -338.8809200, upper bound: 338.8809189
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 21.63
Output dim: 7, lower bound: -338.8809200, upper bound: 338.8809189
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.63
Output dim: 7, lower bound: -338.8872935, upper bound: 338.8872978
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.63
Output dim: 7, lower bound: -338.8872936, upper bound: 338.8872985
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.8870570, upper bound: 338.8869671
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.8869671, upper bound: 338.8870570
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9233753, upper bound: 338.9233777
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9233777, upper bound: 338.9233753
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.8889403, upper bound: 338.8888998
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.8888998, upper bound: 338.8889403
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9217635, upper bound: 338.9217538
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9217635, upper bound: 338.9217538
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9275536, upper bound: 338.9275496
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9275536, upper bound: 338.9275496
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9020448, upper bound: 338.9021090
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9020448, upper bound: 338.9021090
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.8922880, upper bound: 338.8923167
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.8922880, upper bound: 338.8923167
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9026839, upper bound: 338.9026746
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9026839, upper bound: 338.9026746
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9310869, upper bound: 338.9310530
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9310551, upper bound: 338.9310845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9262358, upper bound: 338.9261967
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.9262358, upper bound: 338.9261967
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.8960066, upper bound: 338.8960293
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.63
Output dim: 7, lower bound: -338.8960066, upper bound: 338.8960293
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=341.21295166015625
rel_dist={7: [-338.93443485344073, 338.93443485344073]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9310129, upper bound: 338.9309957
time: 10.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9309957, upper bound: 338.9310129
time: 10.21 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 20.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 20.65
Output dim: 7, lower bound: -338.9310129, upper bound: 338.9309957
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 20.65
Output dim: 7, lower bound: -338.9309957, upper bound: 338.9310129

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
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 238

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9310129, upper bound: 338.9309801
time: 10.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9309988, upper bound: 338.9309957
time: 10.85 seconds

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
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9302532, upper bound: 338.9302606
time: 10.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9302505, upper bound: 338.9302709
time: 9.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.23
Output dim: 7, lower bound: -338.9310129, upper bound: 338.9309801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.23
Output dim: 7, lower bound: -338.9309988, upper bound: 338.9309957
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.23
Output dim: 7, lower bound: -338.9302532, upper bound: 338.9302606
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.23
Output dim: 7, lower bound: -338.9302505, upper bound: 338.9302709

## BFS RS instance: RS_RSZ1_RSZ1

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9086283, upper bound: 338.9086100
time: 9.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9086283, upper bound: 338.9086100
time: 9.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9269386, upper bound: 338.9269366
time: 10.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9269385, upper bound: 338.9269366
time: 9.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9287590, upper bound: 338.9287332
time: 10.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9287590, upper bound: 338.9287332
time: 9.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9067646, upper bound: 338.9067662
time: 9.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9067646, upper bound: 338.9067662
time: 9.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.65
Output dim: 7, lower bound: -338.9086283, upper bound: 338.9086100
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.65
Output dim: 7, lower bound: -338.9086283, upper bound: 338.9086100
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.65
Output dim: 7, lower bound: -338.9269386, upper bound: 338.9269366
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.65
Output dim: 7, lower bound: -338.9269385, upper bound: 338.9269366
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.65
Output dim: 7, lower bound: -338.9287590, upper bound: 338.9287332
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.65
Output dim: 7, lower bound: -338.9287590, upper bound: 338.9287332
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.65
Output dim: 7, lower bound: -338.9067646, upper bound: 338.9067662
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.65
Output dim: 7, lower bound: -338.9067646, upper bound: 338.9067662

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9082291, upper bound: 338.9081873
time: 10.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9082113, upper bound: 338.9082075
time: 9.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9086268, upper bound: 338.9085811
time: 12.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9085891, upper bound: 338.9086038
time: 9.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 162

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9100777, upper bound: 338.9100684
time: 9.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9100777, upper bound: 338.9100684
time: 9.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9263356, upper bound: 338.9263314
time: 9.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9263355, upper bound: 338.9263314
time: 9.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 238

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9287590, upper bound: 338.9287255
time: 10.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9287362, upper bound: 338.9287332
time: 10.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9287590, upper bound: 338.9287326
time: 10.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9287531, upper bound: 338.9287332
time: 10.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 99

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8910795, upper bound: 338.8910943
time: 9.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8910795, upper bound: 338.8910943
time: 9.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9067646, upper bound: 338.9067662
time: 10.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9067639, upper bound: 338.9067662
time: 10.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9082291, upper bound: 338.9081873
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9082113, upper bound: 338.9082075
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9086268, upper bound: 338.9085811
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9085891, upper bound: 338.9086038
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9100777, upper bound: 338.9100684
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9100777, upper bound: 338.9100684
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9263356, upper bound: 338.9263314
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9263355, upper bound: 338.9263314
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9287590, upper bound: 338.9287255
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9287362, upper bound: 338.9287332
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9287590, upper bound: 338.9287326
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9287531, upper bound: 338.9287332
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.8910795, upper bound: 338.8910943
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.8910795, upper bound: 338.8910943
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9067646, upper bound: 338.9067662
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 7, lower bound: -338.9067639, upper bound: 338.9067662

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9082291, upper bound: 338.9081783
time: 10.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9082220, upper bound: 338.9081873
time: 9.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8954792, upper bound: 338.8954875
time: 9.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8954792, upper bound: 338.8954875
time: 11.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9082340, upper bound: 338.9082248
time: 9.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9082464, upper bound: 338.9082248
time: 9.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9060551, upper bound: 338.9060535
time: 9.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9060551, upper bound: 338.9060535
time: 9.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9087496, upper bound: 338.9087497
time: 9.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9087480, upper bound: 338.9087528
time: 9.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9003273, upper bound: 338.9003358
time: 9.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9003273, upper bound: 338.9003358
time: 9.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9091818, upper bound: 338.9091881
time: 10.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9091818, upper bound: 338.9091881
time: 10.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8890428, upper bound: 338.8890691
time: 9.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8890428, upper bound: 338.8890691
time: 8.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9287351, upper bound: 338.9286902
time: 9.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9287129, upper bound: 338.9287178
time: 9.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9279055, upper bound: 338.9279116
time: 10.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9279055, upper bound: 338.9279116
time: 10.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 238

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9287590, upper bound: 338.9287242
time: 10.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9287362, upper bound: 338.9287326
time: 10.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 162

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9115307, upper bound: 338.9115289
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9115307, upper bound: 338.9115289
time: 9.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8798567, upper bound: 338.8798701
time: 8.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8798567, upper bound: 338.8798701
time: 8.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8739674, upper bound: 338.8739864
time: 10.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8739674, upper bound: 338.8739864
time: 9.51 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9082291, upper bound: 338.9081783
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9082220, upper bound: 338.9081873
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.8954792, upper bound: 338.8954875
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.8954792, upper bound: 338.8954875
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9082340, upper bound: 338.9082248
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9082464, upper bound: 338.9082248
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9060551, upper bound: 338.9060535
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9060551, upper bound: 338.9060535
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9087496, upper bound: 338.9087497
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9087480, upper bound: 338.9087528
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9003273, upper bound: 338.9003358
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9003273, upper bound: 338.9003358
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9091818, upper bound: 338.9091881
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9091818, upper bound: 338.9091881
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.8890428, upper bound: 338.8890691
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.8890428, upper bound: 338.8890691
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9287351, upper bound: 338.9286902
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9287129, upper bound: 338.9287178
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9279055, upper bound: 338.9279116
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9279055, upper bound: 338.9279116
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9287590, upper bound: 338.9287242
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9287362, upper bound: 338.9287326
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9115307, upper bound: 338.9115289
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.9115307, upper bound: 338.9115289
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.8798567, upper bound: 338.8798701
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.8798567, upper bound: 338.8798701
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.8739674, upper bound: 338.8739864
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.69
Output dim: 7, lower bound: -338.8739674, upper bound: 338.8739864
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.69
Output dim: 7, lower bound: -338.9067646, upper bound: 338.9067662
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.69
Output dim: 7, lower bound: -338.9067639, upper bound: 338.9067662
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=341.21295166015625
rel_dist={7: [-338.93400199054076, 338.93400199054076]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9233461, upper bound: 338.9233461
time: 11.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9233461, upper bound: 338.9233461
time: 16.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 27.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 27.88
Output dim: 7, lower bound: -338.9233461, upper bound: 338.9233461
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 27.88
Output dim: 7, lower bound: -338.9233461, upper bound: 338.9233461

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8991685, upper bound: 338.8991685
time: 12.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8991685, upper bound: 338.8991685
time: 10.95 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9053609, upper bound: 338.9053674
time: 12.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9053609, upper bound: 338.9053674
time: 11.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.59
Output dim: 7, lower bound: -338.8991685, upper bound: 338.8991685
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.59
Output dim: 7, lower bound: -338.8991685, upper bound: 338.8991685
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.59
Output dim: 7, lower bound: -338.9053609, upper bound: 338.9053674
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.59
Output dim: 7, lower bound: -338.9053609, upper bound: 338.9053674

## BFS RS instance: RS_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8991684, upper bound: 338.8991685
time: 10.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8991685, upper bound: 338.8991684
time: 11.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8880524, upper bound: 338.8880532
time: 10.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8880524, upper bound: 338.8880524
time: 10.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8907179, upper bound: 338.8907179
time: 10.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8907179, upper bound: 338.8907179
time: 10.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9053597, upper bound: 338.9053674
time: 11.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9053609, upper bound: 338.9053671
time: 12.13 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -338.8991684, upper bound: 338.8991685
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -338.8991685, upper bound: 338.8991684
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -338.8880524, upper bound: 338.8880532
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -338.8880524, upper bound: 338.8880524
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -338.8907179, upper bound: 338.8907179
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -338.8907179, upper bound: 338.8907179
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -338.9053597, upper bound: 338.9053674
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -338.9053609, upper bound: 338.9053671

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8991684, upper bound: 338.8991643
time: 13.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8991643, upper bound: 338.8991685
time: 12.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8979220, upper bound: 338.8979222
time: 11.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8979222, upper bound: 338.8979220
time: 11.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874387, upper bound: 338.8874364
time: 9.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874348, upper bound: 338.8874395
time: 9.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8880532, upper bound: 338.8880498
time: 9.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8880505, upper bound: 338.8880524
time: 10.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 99

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8907174, upper bound: 338.8907179
time: 10.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8907174, upper bound: 338.8907174
time: 10.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8880748, upper bound: 338.8880748
time: 13.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8880748, upper bound: 338.8880748
time: 10.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8908784, upper bound: 338.8908803
time: 10.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8908784, upper bound: 338.8908803
time: 10.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8886054, upper bound: 338.8886055
time: 12.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8886054, upper bound: 338.8886055
time: 11.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8991684, upper bound: 338.8991643
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8991643, upper bound: 338.8991685
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8979220, upper bound: 338.8979222
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8979222, upper bound: 338.8979220
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8874387, upper bound: 338.8874364
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8874348, upper bound: 338.8874395
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8880532, upper bound: 338.8880498
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8880505, upper bound: 338.8880524
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8907174, upper bound: 338.8907179
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8907174, upper bound: 338.8907174
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8880748, upper bound: 338.8880748
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8880748, upper bound: 338.8880748
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8908784, upper bound: 338.8908803
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8908784, upper bound: 338.8908803
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8886054, upper bound: 338.8886055
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.85
Output dim: 7, lower bound: -338.8886054, upper bound: 338.8886055

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8991683, upper bound: 338.8991643
time: 10.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8991684, upper bound: 338.8991642
time: 12.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8985587, upper bound: 338.8985635
time: 9.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8985587, upper bound: 338.8985635
time: 12.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8606774, upper bound: 338.8606777
time: 9.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8606774, upper bound: 338.8606777
time: 9.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8818699, upper bound: 338.8818737
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8818699, upper bound: 338.8818737
time: 9.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8742961, upper bound: 338.8742915
time: 12.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8742961, upper bound: 338.8742915
time: 12.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874348, upper bound: 338.8874340
time: 10.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874294, upper bound: 338.8874395
time: 9.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8825759, upper bound: 338.8825757
time: 12.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8825759, upper bound: 338.8825757
time: 12.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 99
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8851518, upper bound: 338.8851517
time: 12.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8851517, upper bound: 338.8851522
time: 9.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8903404, upper bound: 338.8903345
time: 10.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8903343, upper bound: 338.8903409
time: 9.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8907142, upper bound: 338.8907174
time: 13.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8907179, upper bound: 338.8907142
time: 9.78 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8991683, upper bound: 338.8991643
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8991684, upper bound: 338.8991642
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8985587, upper bound: 338.8985635
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8985587, upper bound: 338.8985635
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8606774, upper bound: 338.8606777
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8606774, upper bound: 338.8606777
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8818699, upper bound: 338.8818737
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8818699, upper bound: 338.8818737
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8742961, upper bound: 338.8742915
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8742961, upper bound: 338.8742915
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8874348, upper bound: 338.8874340
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8874294, upper bound: 338.8874395
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8825759, upper bound: 338.8825757
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8825759, upper bound: 338.8825757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8851518, upper bound: 338.8851517
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8851517, upper bound: 338.8851522
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8903404, upper bound: 338.8903345
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8903343, upper bound: 338.8903409
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8907142, upper bound: 338.8907174
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 7, lower bound: -338.8907179, upper bound: 338.8907142
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 7, lower bound: -338.8880748, upper bound: 338.8880748
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 7, lower bound: -338.8880748, upper bound: 338.8880748
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 7, lower bound: -338.8908784, upper bound: 338.8908803
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 7, lower bound: -338.8908784, upper bound: 338.8908803
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 7, lower bound: -338.8886054, upper bound: 338.8886055
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.11
Output dim: 7, lower bound: -338.8886054, upper bound: 338.8886055
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=341.21295166015625
rel_dist={7: [-338.93322554933513, 338.933225507835]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1842.85 seconds
