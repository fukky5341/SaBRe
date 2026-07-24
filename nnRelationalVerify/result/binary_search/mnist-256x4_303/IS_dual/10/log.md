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
execution time: IAR + LP analysis = 1.15 + 11.70 = 12.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -338.9345818, upper bound: 338.9345818


# Binary Search by BASE starts (time budget: 2687.15 seconds, max iter: 100)

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
Binary search time: 44.88 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2642.28 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9148962, upper bound: 338.9145436
time: 10.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9073139, upper bound: 338.9073139
time: 10.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.07 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.07
Output dim: 7, lower bound: -338.9148962, upper bound: 338.9145436
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.07
Output dim: 7, lower bound: -338.9073139, upper bound: 338.9073139

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -184.7211609, 146.8837128, -328.2977295, 328.9904175
1: -152.3174133, 128.1564331, -155.1049194, 130.5072479, -282.8246460, 283.2612915
2: -200.0067291, 129.6382141, -203.6430969, 131.9900055, -331.9967346, 333.2812805
3: -212.3461914, 112.1826477, -216.2613983, 114.2387161, -326.5848389, 328.4439697
4: -194.8282928, 148.9458313, -198.3916016, 151.6778870, -346.5061646, 347.3374329
5: -174.5253906, 135.6783600, -177.7157135, 138.1587830, -312.6841736, 313.3940430
6: -167.2519073, 161.0517578, -170.2996063, 163.9844208, -331.2363281, 331.3513794
7: -181.8564148, 153.2529755, -185.1827240, 156.0302124, -337.8866272, 338.4356995
8: -219.8477783, 150.3132477, -223.8251801, 153.0471344, -372.8948975, 374.1384277
9: -166.0647430, 163.1612549, -169.0843506, 166.1456757, -332.2103577, 332.2455139

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9036497, upper bound: 338.9043945
time: 11.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8863462, upper bound: 338.8855537
time: 8.82 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -180.5725403, 143.6135406, -182.1839142, 144.8766632, -325.4491882, 325.7974548
1: -151.4880676, 127.4342117, -152.9629974, 128.7038727, -280.1918335, 280.3971863
2: -199.0630951, 129.0347137, -200.8489227, 130.1867065, -329.2497864, 329.8836365
3: -211.2796173, 111.5729218, -213.2581177, 112.6657028, -323.9453125, 324.8309937
4: -193.8635254, 148.0949554, -195.6531372, 149.5814514, -343.4449463, 343.7480774
5: -173.5881195, 134.9150391, -175.2618561, 136.2559204, -309.8439941, 310.1768799
6: -166.5021667, 160.2713776, -167.9635468, 161.7322693, -328.2344360, 328.2348633
7: -180.9860382, 152.5078735, -182.6294861, 153.8943481, -334.8803406, 335.1373291
8: -218.9145355, 149.4556274, -220.7750397, 150.9503326, -369.8648071, 370.2306519
9: -165.2428284, 162.2864838, -166.7643890, 163.8521118, -329.0949402, 329.0508423

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8809748, upper bound: 338.8822423
time: 10.64 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8733966, upper bound: 338.8733966
time: 7.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.51 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.51
Output dim: 7, lower bound: -338.9036497, upper bound: 338.9043945
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 19.51
Output dim: 7, lower bound: -338.8863462, upper bound: 338.8855537
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 19.51
Output dim: 7, lower bound: -338.8809748, upper bound: 338.8822423
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 19.51
Output dim: 7, lower bound: -338.8733966, upper bound: 338.8733966

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -181.8114929, 144.5773010, -325.9913330, 326.0807495
1: -152.3174133, 128.1564331, -152.6432495, 128.4476624, -280.7650757, 280.7996216
2: -200.0067291, 129.6382141, -200.4200897, 129.9217987, -329.9285278, 330.0582886
3: -212.3461914, 112.1826477, -212.8362732, 112.4366455, -324.7827454, 325.0188599
4: -194.8282928, 148.9458313, -195.2288971, 149.2648773, -344.0931702, 344.1747437
5: -174.5253906, 135.6783600, -174.8884888, 135.9642639, -310.4896545, 310.5668335
6: -167.2519073, 161.0517578, -167.6027069, 161.3926849, -328.6445923, 328.6544800
7: -181.8564148, 153.2529755, -182.2293854, 153.5557556, -335.4121704, 335.4823608
8: -219.8477783, 150.3132477, -220.3479004, 150.6572723, -370.5050659, 370.6611328
9: -166.0647430, 163.1612549, -166.3887329, 163.5004578, -329.5650940, 329.5499573

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8844753, upper bound: 338.8862881
time: 10.85 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8892236, upper bound: 338.8895297
time: 10.59 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8728921, upper bound: 338.8714903
time: 9.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 42.68 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 42.68
Output dim: 7, lower bound: -338.8892236, upper bound: 338.8895297
IS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 42.68
Output dim: 7, lower bound: -338.8728921, upper bound: 338.8714903

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -172.1764526, 136.9344330, -318.3484192, 316.4456787
1: -152.3174133, 128.1564331, -144.5820465, 121.6016312, -273.9190369, 272.7384644
2: -200.0067291, 129.6382141, -189.8091583, 123.1007156, -323.1074524, 319.4473877
3: -212.3461914, 112.1826477, -201.4711914, 106.4826202, -318.8287659, 313.6538391
4: -194.8282928, 148.9458313, -184.8327942, 141.3356018, -336.1638794, 333.7786255
5: -174.5253906, 135.6783600, -165.5833740, 128.7392578, -303.2646179, 301.2616577
6: -167.2519073, 161.0517578, -158.7264252, 152.8542480, -320.1061401, 319.7781982
7: -181.8564148, 153.2529755, -172.5192719, 145.4586487, -327.3150635, 325.7722168
8: -219.8477783, 150.3132477, -208.8358765, 142.6981049, -362.5458984, 359.1491089
9: -166.0647430, 163.1612549, -157.5827789, 154.7900085, -320.8547058, 320.7439575

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8708674, upper bound: 338.8719116
time: 10.41 seconds

## Relational analysis of IS_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8728921, upper bound: 338.8714903
time: 10.18 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8728921, upper bound: 338.8714903
time: 10.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 42.07 seconds
IS_A1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 42.07
Output dim: 7, lower bound: -338.8728921, upper bound: 338.8714903
IS_A1_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 42.07
Output dim: 7, lower bound: -338.8728921, upper bound: 338.8714903
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=341.21295166015625
rel_dist={7: [-338.93443485344073, 338.93443485344073]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9177246, upper bound: 338.9171699
time: 8.93 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9074359, upper bound: 338.9074359
time: 8.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.72
Output dim: 7, lower bound: -338.9177246, upper bound: 338.9171699
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.72
Output dim: 7, lower bound: -338.9074359, upper bound: 338.9074359

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -184.7211609, 146.8837128, -328.2977295, 328.9904175
1: -152.3174133, 128.1564331, -155.1049194, 130.5072479, -282.8246460, 283.2612915
2: -200.0067291, 129.6382141, -203.6430969, 131.9900055, -331.9967346, 333.2812805
3: -212.3461914, 112.1826477, -216.2613983, 114.2387161, -326.5848389, 328.4439697
4: -194.8282928, 148.9458313, -198.3916016, 151.6778870, -346.5061646, 347.3374329
5: -174.5253906, 135.6783600, -177.7157135, 138.1587830, -312.6841736, 313.3940430
6: -167.2519073, 161.0517578, -170.2996063, 163.9844208, -331.2363281, 331.3513794
7: -181.8564148, 153.2529755, -185.1827240, 156.0302124, -337.8866272, 338.4356995
8: -219.8477783, 150.3132477, -223.8251801, 153.0471344, -372.8948975, 374.1384277
9: -166.0647430, 163.1612549, -169.0843506, 166.1456757, -332.2103577, 332.2455139

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9091634, upper bound: 338.9095691
time: 9.80 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8918637, upper bound: 338.8906303
time: 8.89 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -180.5725403, 143.6135406, -183.7982178, 146.1535950, -326.7261353, 327.4117432
1: -151.4880676, 127.4342117, -154.3257141, 129.8512726, -281.3392334, 281.7598877
2: -199.0630951, 129.0347137, -202.6266479, 131.3341522, -330.3972473, 331.6613464
3: -211.2796173, 111.5729218, -215.1687775, 113.6665268, -324.9461365, 326.7416687
4: -193.8635254, 148.0949554, -197.3954315, 150.9152985, -344.7788086, 345.4903870
5: -173.5881195, 134.9150391, -176.8230896, 137.4666290, -311.0547485, 311.7381287
6: -166.5021667, 160.2713776, -169.4499207, 163.1651459, -329.6672974, 329.7212219
7: -180.9860382, 152.5078735, -184.2539978, 155.2533112, -336.2393494, 336.7618408
8: -218.9145355, 149.4556274, -222.7155457, 152.2846375, -371.1990662, 372.1711731
9: -165.2428284, 162.2864838, -168.2404785, 165.3114014, -330.5542297, 330.5269470

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8840284, upper bound: 338.8858064
time: 7.92 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8734698, upper bound: 338.8734698
time: 7.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.37
Output dim: 7, lower bound: -338.9091634, upper bound: 338.9095691
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.37
Output dim: 7, lower bound: -338.8918637, upper bound: 338.8906303
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 16.37
Output dim: 7, lower bound: -338.8840284, upper bound: 338.8858064
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 16.37
Output dim: 7, lower bound: -338.8734698, upper bound: 338.8734698

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -181.8114929, 144.5773010, -325.9913330, 326.0807495
1: -152.3174133, 128.1564331, -152.6432495, 128.4476624, -280.7650757, 280.7996216
2: -200.0067291, 129.6382141, -200.4200897, 129.9217987, -329.9285278, 330.0582886
3: -212.3461914, 112.1826477, -212.8362732, 112.4366455, -324.7827454, 325.0188599
4: -194.8282928, 148.9458313, -195.2288971, 149.2648773, -344.0931702, 344.1747437
5: -174.5253906, 135.6783600, -174.8884888, 135.9642639, -310.4896545, 310.5668335
6: -167.2519073, 161.0517578, -167.6027069, 161.3926849, -328.6445923, 328.6544800
7: -181.8564148, 153.2529755, -182.2293854, 153.5557556, -335.4121704, 335.4823608
8: -219.8477783, 150.3132477, -220.3479004, 150.6572723, -370.5050659, 370.6611328
9: -166.0647430, 163.1612549, -166.3887329, 163.5004578, -329.5650940, 329.5499573

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8951332, upper bound: 338.8970508
time: 9.01 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8984120, upper bound: 338.8988549
time: 9.93 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8833535, upper bound: 338.8821803
time: 9.43 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -181.1428833, 144.0546417, -187.0249176, 148.7090454, -329.8519287, 331.0794983
1: -152.0875854, 127.9644623, -156.9591064, 132.0491638, -284.1367493, 284.9235840
2: -199.7066345, 129.4449921, -206.1374054, 133.6064758, -333.3131104, 335.5823975
3: -212.0265198, 112.0149384, -218.9130402, 115.6342392, -327.6607361, 330.9279175
4: -194.5341644, 148.7213440, -200.7862701, 153.4700317, -348.0042114, 349.5075684
5: -174.2619934, 135.4739532, -179.8272095, 139.7757721, -314.0377502, 315.3011475
6: -167.0007019, 160.8107300, -172.3595581, 165.9728241, -332.9734802, 333.1702881
7: -181.5814362, 153.0227051, -187.4101410, 157.8600922, -339.4414978, 340.4328613
8: -219.5230408, 150.0915680, -226.6953888, 154.9080811, -374.4311218, 376.7869568
9: -165.8144226, 162.9151306, -171.0719910, 168.0565186, -333.8709106, 333.9870911

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8745957, upper bound: 338.8759455
time: 7.41 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8800795, upper bound: 338.8788458
time: 9.20 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8803179, upper bound: 338.8801391
time: 9.08 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8634938, upper bound: 338.8647374
time: 10.04 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8822795, upper bound: 338.8815690
time: 8.88 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8918637, upper bound: 338.8906303
time: 10.07 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 79.97 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 79.97
Output dim: 7, lower bound: -338.8984120, upper bound: 338.8988549
IS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 79.97
Output dim: 7, lower bound: -338.8833535, upper bound: 338.8821803
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 79.97
Output dim: 7, lower bound: -338.8822795, upper bound: 338.8815690
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 79.97
Output dim: 7, lower bound: -338.8918637, upper bound: 338.8906303

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -172.1764526, 136.9344330, -318.3484192, 316.4456787
1: -152.3174133, 128.1564331, -144.5820465, 121.6016312, -273.9190369, 272.7384644
2: -200.0067291, 129.6382141, -189.8091583, 123.1007156, -323.1074524, 319.4473877
3: -212.3461914, 112.1826477, -201.4711914, 106.4826202, -318.8287659, 313.6538391
4: -194.8282928, 148.9458313, -184.8327942, 141.3356018, -336.1638794, 333.7786255
5: -174.5253906, 135.6783600, -165.5833740, 128.7392578, -303.2646179, 301.2616577
6: -167.2519073, 161.0517578, -158.7264252, 152.8542480, -320.1061401, 319.7781982
7: -181.8564148, 153.2529755, -172.5192719, 145.4586487, -327.3150635, 325.7722168
8: -219.8477783, 150.3132477, -208.8358765, 142.6981049, -362.5458984, 359.1491089
9: -166.0647430, 163.1612549, -157.5827789, 154.7900085, -320.8547058, 320.7439575

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8850319, upper bound: 338.8864769
time: 9.90 seconds

## Relational analysis of IS_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8754536, upper bound: 338.8736861
time: 9.49 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8719683, upper bound: 338.8702991
time: 9.73 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -180.2300720, 143.3327942, -187.0249176, 148.7090454, -328.9391174, 330.3576965
1: -151.3177948, 127.3191223, -156.9591064, 132.0491638, -283.3669434, 284.2782288
2: -198.6952362, 128.7892914, -206.1374054, 133.6064758, -332.3016968, 334.9266968
3: -210.9494476, 111.4491959, -218.9130402, 115.6342392, -326.5836182, 330.3621826
4: -193.5493164, 147.9714966, -200.7862701, 153.4700317, -347.0193481, 348.7576904
5: -173.3876190, 134.7928009, -179.8272095, 139.7757721, -313.1633301, 314.6199951
6: -166.1574554, 159.9978027, -172.3595581, 165.9728241, -332.1302795, 332.3573608
7: -180.6571198, 152.2518921, -187.4101410, 157.8600922, -338.5171509, 339.6620483
8: -218.4125519, 149.3382416, -226.6953888, 154.9080811, -373.3206177, 376.0336304
9: -164.9800110, 162.0939484, -171.0719910, 168.0565186, -333.0364685, 333.1658936

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8745957, upper bound: 338.8759455
time: 8.22 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8800795, upper bound: 338.8788458
time: 8.73 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8803179, upper bound: 338.8801391
time: 8.73 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8634938, upper bound: 338.8647374
time: 7.65 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8842721, upper bound: 338.8835222
time: 8.99 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8897027, upper bound: 338.8879831
time: 8.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 82.23 seconds
IS_A1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 82.23
Output dim: 7, lower bound: -338.8754536, upper bound: 338.8736861
IS_A1_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 82.23
Output dim: 7, lower bound: -338.8719683, upper bound: 338.8702991
IS_A1_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 82.23
Output dim: 7, lower bound: -338.8842721, upper bound: 338.8835222
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 82.23
Output dim: 7, lower bound: -338.8897027, upper bound: 338.8879831

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -177.4806366, 141.1811218, -187.0249176, 148.7090454, -326.1896973, 328.2059937
1: -149.0117645, 125.3912735, -156.9591064, 132.0491638, -281.0609131, 282.3503418
2: -195.6909485, 126.8320694, -206.1374054, 133.6064758, -329.2974243, 332.9694824
3: -207.7441406, 109.7544250, -218.9130402, 115.6342392, -323.3783569, 328.6674500
4: -190.6048584, 145.7005310, -200.7862701, 153.4700317, -344.0748901, 346.4867554
5: -170.7597656, 132.7411957, -179.8272095, 139.7757721, -310.5354309, 312.5684204
6: -163.6218872, 157.5812378, -172.3595581, 165.9728241, -329.5947266, 329.9407959
7: -177.9108734, 149.9645538, -187.4101410, 157.8600922, -335.7709045, 337.3746948
8: -215.1028290, 147.0612946, -226.6953888, 154.9080811, -370.0108948, 373.7566833
9: -162.4847107, 159.6208649, -171.0719910, 168.0565186, -330.5411987, 330.6928711

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8720492, upper bound: 338.8731540
time: 9.60 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8777775, upper bound: 338.8762063
time: 9.21 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8781573, upper bound: 338.8776738
time: 8.76 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8609724, upper bound: 338.8621735
time: 8.44 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8706905, upper bound: 338.8704467
time: 7.97 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8897027, upper bound: 338.8879831
time: 9.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 83.24 seconds
IS_A1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 83.24
Output dim: 7, lower bound: -338.8706905, upper bound: 338.8704467
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 83.24
Output dim: 7, lower bound: -338.8897027, upper bound: 338.8879831

## BFS IS instance: IS_A1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -177.0151978, 140.8103180, -187.0249176, 148.7090454, -325.7242432, 327.8352356
1: -148.6207581, 125.0639954, -156.9591064, 132.0491638, -280.6699219, 282.0231018
2: -195.1765137, 126.5002441, -206.1374054, 133.6064758, -328.7829895, 332.6376343
3: -207.1953888, 109.4676056, -218.9130402, 115.6342392, -322.8296204, 328.3806458
4: -190.1038818, 145.3187561, -200.7862701, 153.4700317, -343.5739136, 346.1049805
5: -170.3109283, 132.3941498, -179.8272095, 139.7757721, -310.0866089, 312.2213745
6: -163.1929932, 157.1683960, -172.3595581, 165.9728241, -329.1658325, 329.5279541
7: -177.4422607, 149.5712891, -187.4101410, 157.8600922, -335.3023071, 336.9814148
8: -214.5399323, 146.6796875, -226.6953888, 154.9080811, -369.4479370, 373.3750610
9: -162.0595245, 159.2026825, -171.0719910, 168.0565186, -330.1160278, 330.2746582

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8720492, upper bound: 338.8731540
time: 9.98 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8777775, upper bound: 338.8762063
time: 9.63 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8781573, upper bound: 338.8776738
time: 10.31 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8609724, upper bound: 338.8621735
time: 9.42 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8794328, upper bound: 338.8776997
time: 9.06 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8807468, upper bound: 338.8791636
time: 8.78 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8807688, upper bound: 338.8791875
time: 8.47 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 108.60 seconds
IS_A1_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 108.60
Output dim: 7, lower bound: -338.8807468, upper bound: 338.8791636
IS_A1_B2_A2_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 108.60
Output dim: 7, lower bound: -338.8807688, upper bound: 338.8791875
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=341.21295166015625
rel_dist={7: [-338.93451046142036, 338.9345104226113]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9191064, upper bound: 338.9185817
time: 8.62 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9075016, upper bound: 338.9075016
time: 8.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.79
Output dim: 7, lower bound: -338.9191064, upper bound: 338.9185817
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.79
Output dim: 7, lower bound: -338.9075016, upper bound: 338.9075016

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -184.7211609, 146.8837128, -328.2977295, 328.9904175
1: -152.3174133, 128.1564331, -155.1049194, 130.5072479, -282.8246460, 283.2612915
2: -200.0067291, 129.6382141, -203.6430969, 131.9900055, -331.9967346, 333.2812805
3: -212.3461914, 112.1826477, -216.2613983, 114.2387161, -326.5848389, 328.4439697
4: -194.8282928, 148.9458313, -198.3916016, 151.6778870, -346.5061646, 347.3374329
5: -174.5253906, 135.6783600, -177.7157135, 138.1587830, -312.6841736, 313.3940430
6: -167.2519073, 161.0517578, -170.2996063, 163.9844208, -331.2363281, 331.3513794
7: -181.8564148, 153.2529755, -185.1827240, 156.0302124, -337.8866272, 338.4356995
8: -219.8477783, 150.3132477, -223.8251801, 153.0471344, -372.8948975, 374.1384277
9: -166.0647430, 163.1612549, -169.0843506, 166.1456757, -332.2103577, 332.2455139

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9117379, upper bound: 338.9120676
time: 9.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8952544, upper bound: 338.8936844
time: 9.05 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -180.5725403, 143.6135406, -184.4727173, 146.6871948, -327.2597046, 328.0862427
1: -151.4880676, 127.4342117, -154.8951569, 130.3306580, -281.8186646, 282.3293457
2: -199.0630951, 129.0347137, -203.3694916, 131.8134766, -330.8765869, 332.4041748
3: -211.2796173, 111.5729218, -215.9672394, 114.0847015, -325.3643188, 327.5401611
4: -193.8635254, 148.0949554, -198.1234741, 151.4725952, -345.3360901, 346.2184143
5: -173.5881195, 134.9150391, -177.4754486, 137.9724579, -311.5605774, 312.3904724
6: -166.5021667, 160.2713776, -170.0708923, 163.7639008, -330.2660522, 330.3422546
7: -180.9860382, 152.5078735, -184.9327545, 155.8210754, -336.8071289, 337.4406128
8: -218.9145355, 149.4556274, -223.5265198, 152.8418579, -371.7563171, 372.9820557
9: -165.2428284, 162.2864838, -168.8572235, 165.9211273, -331.1639404, 331.1437073

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8858829, upper bound: 338.8878754
time: 9.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8735140, upper bound: 338.8735140
time: 6.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.01 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.01
Output dim: 7, lower bound: -338.9117379, upper bound: 338.9120676
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.01
Output dim: 7, lower bound: -338.8952544, upper bound: 338.8936844
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.01
Output dim: 7, lower bound: -338.8858829, upper bound: 338.8878754
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 17.01
Output dim: 7, lower bound: -338.8735140, upper bound: 338.8735140

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -181.8114929, 144.5773010, -325.9913330, 326.0807495
1: -152.3174133, 128.1564331, -152.6432495, 128.4476624, -280.7650757, 280.7996216
2: -200.0067291, 129.6382141, -200.4200897, 129.9217987, -329.9285278, 330.0582886
3: -212.3461914, 112.1826477, -212.8362732, 112.4366455, -324.7827454, 325.0188599
4: -194.8282928, 148.9458313, -195.2288971, 149.2648773, -344.0931702, 344.1747437
5: -174.5253906, 135.6783600, -174.8884888, 135.9642639, -310.4896545, 310.5668335
6: -167.2519073, 161.0517578, -167.6027069, 161.3926849, -328.6445923, 328.6544800
7: -181.8564148, 153.2529755, -182.2293854, 153.5557556, -335.4121704, 335.4823608
8: -219.8477783, 150.3132477, -220.3479004, 150.6572723, -370.5050659, 370.6611328
9: -166.0647430, 163.1612549, -166.3887329, 163.5004578, -329.5650940, 329.5499573

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8996322, upper bound: 338.9015826
time: 10.46 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9020905, upper bound: 338.9028457
time: 9.16 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8881866, upper bound: 338.8870273
time: 10.18 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -187.0249176, 148.7090454, -330.1231079, 331.2941589
1: -152.3174133, 128.1564331, -156.9591064, 132.0491638, -284.3665771, 285.1155090
2: -200.0067291, 129.6382141, -206.1374054, 133.6064758, -333.6132202, 335.7756348
3: -212.3461914, 112.1826477, -218.9130402, 115.6342392, -327.9803162, 331.0956421
4: -194.8282928, 148.9458313, -200.7862701, 153.4700317, -348.2983398, 349.7320862
5: -174.5253906, 135.6783600, -179.8272095, 139.7757721, -314.3011169, 315.5055542
6: -167.2519073, 161.0517578, -172.3595581, 165.9728241, -333.2247314, 333.4113159
7: -181.8564148, 153.2529755, -187.4101410, 157.8600922, -339.7164917, 340.6631165
8: -219.8477783, 150.3132477, -226.6953888, 154.9080811, -374.7558594, 377.0086060
9: -166.0647430, 163.1612549, -171.0719910, 168.0565186, -334.1211853, 334.2331848

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8813169, upper bound: 338.8822467
time: 8.70 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8850038, upper bound: 338.8835604
time: 8.60 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8668827, upper bound: 338.8684561
time: 8.93 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8864774, upper bound: 338.8861433
time: 9.50 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8744282, upper bound: 338.8752597
time: 8.66 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8890360, upper bound: 338.8878281
time: 9.66 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8952545, upper bound: 338.8936844
time: 8.41 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -180.5725403, 143.6135406, -181.5639343, 144.3814240, -324.9539795, 325.1774902
1: -151.4880676, 127.4342117, -152.4342651, 128.2716980, -279.7597046, 279.8683777
2: -199.0630951, 129.0347137, -200.1474457, 129.7458801, -328.8089600, 329.1821594
3: -211.2796173, 111.5729218, -212.5431976, 112.2831573, -323.5627747, 324.1160889
4: -193.8635254, 148.0949554, -194.9616699, 149.0603027, -342.9238281, 343.0566101
5: -173.5881195, 134.9150391, -174.6490479, 135.7786407, -309.3667297, 309.5640869
6: -166.5021667, 160.2713776, -167.3747711, 161.1729584, -327.6751099, 327.6461487
7: -180.9860382, 152.5078735, -181.9802399, 153.3473969, -334.3334351, 334.4880981
8: -218.9145355, 149.4556274, -220.0502777, 150.4526520, -369.3671265, 369.5058899
9: -165.2428284, 162.2864838, -166.1623535, 163.2767029, -328.5194702, 328.4488525

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8629858, upper bound: 338.8630604
time: 7.12 seconds

## Relational analysis of IS_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8629810, upper bound: 338.8628430
time: 7.94 seconds

## Relational analysis of IS_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8724928, upper bound: 338.8731256
time: 9.05 seconds

## Relational analysis of IS_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8738571, upper bound: 338.8753476
time: 8.92 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8835645, upper bound: 338.8860315
time: 9.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 63.68 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 63.68
Output dim: 7, lower bound: -338.9020905, upper bound: 338.9028457
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 63.68
Output dim: 7, lower bound: -338.8881866, upper bound: 338.8870273
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 63.68
Output dim: 7, lower bound: -338.8890360, upper bound: 338.8878281
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 63.68
Output dim: 7, lower bound: -338.8952545, upper bound: 338.8936844
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 63.68
Output dim: 7, lower bound: -338.8738571, upper bound: 338.8753476
IS_A2_B1_B2, status: Status.VERIFIED, split count: 3, time: 63.68
Output dim: 7, lower bound: -338.8835645, upper bound: 338.8860315

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -172.1764526, 136.9344330, -318.3484192, 316.4456787
1: -152.3174133, 128.1564331, -144.5820465, 121.6016312, -273.9190369, 272.7384644
2: -200.0067291, 129.6382141, -189.8091583, 123.1007156, -323.1074524, 319.4473877
3: -212.3461914, 112.1826477, -201.4711914, 106.4826202, -318.8287659, 313.6538391
4: -194.8282928, 148.9458313, -184.8327942, 141.3356018, -336.1638794, 333.7786255
5: -174.5253906, 135.6783600, -165.5833740, 128.7392578, -303.2646179, 301.2616577
6: -167.2519073, 161.0517578, -158.7264252, 152.8542480, -320.1061401, 319.7781982
7: -181.8564148, 153.2529755, -172.5192719, 145.4586487, -327.3150635, 325.7722168
8: -219.8477783, 150.3132477, -208.8358765, 142.6981049, -362.5458984, 359.1491089
9: -166.0647430, 163.1612549, -157.5827789, 154.7900085, -320.8547058, 320.7439575

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8906424, upper bound: 338.8922113
time: 9.49 seconds

## Relational analysis of IS_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8839232, upper bound: 338.8820944
time: 9.83 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8802510, upper bound: 338.8788602
time: 9.34 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -171.2836151, 136.2437592, -317.6578064, 315.5528564
1: -152.3174133, 128.1564331, -143.7672577, 120.9179611, -273.2353821, 271.9237061
2: -200.0067291, 129.6382141, -188.7852478, 122.4580688, -322.4647827, 318.4234619
3: -212.3461914, 112.1826477, -200.3978882, 105.9651566, -318.3113098, 312.5804749
4: -194.8282928, 148.9458313, -183.8282013, 140.5602417, -335.3885498, 332.7740479
5: -174.5253906, 135.6783600, -164.7034912, 128.0302429, -302.5556030, 300.3818359
6: -167.2519073, 161.0517578, -157.8869171, 152.0559998, -319.3079224, 318.9386292
7: -181.8564148, 153.2529755, -171.5707855, 144.7081909, -326.5646057, 324.8237610
8: -219.8477783, 150.3132477, -207.8051453, 141.9419098, -361.7896729, 358.1184082
9: -166.0647430, 163.1612549, -156.7622375, 153.9296265, -319.9942627, 319.9234009

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8753209, upper bound: 338.8753253
time: 10.69 seconds

## Relational analysis of IS_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8571714, upper bound: 338.8571401
time: 9.55 seconds

## Relational analysis of IS_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8805714, upper bound: 338.8796582
time: 9.20 seconds

## Relational analysis of IS_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8620092, upper bound: 338.8624343
time: 9.91 seconds

## Relational analysis of IS_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8816923, upper bound: 338.8808662
time: 8.89 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8859994, upper bound: 338.8844409
time: 9.94 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -187.8389130, 149.3052673, -187.0194550, 148.7047272, -336.5436096, 336.3247070
1: -157.7123718, 132.5779114, -156.9544983, 132.0453033, -289.7576904, 289.5323486
2: -206.9463043, 134.1416321, -206.1313629, 133.6025543, -340.5488586, 340.2729797
3: -219.7465973, 115.9830017, -218.9066010, 115.6308594, -335.3774414, 334.8895874
4: -201.7513428, 154.1855011, -200.7803497, 153.4655457, -355.2168884, 354.9658203
5: -180.7162476, 140.4382629, -179.8219147, 139.7716980, -320.4879150, 320.2601929
6: -173.1458435, 166.6222687, -172.3545074, 165.9679413, -339.1137695, 338.9767761
7: -188.1566620, 158.5348816, -187.4045868, 157.8554840, -346.0121155, 345.9394531
8: -227.4987946, 155.6022034, -226.6887207, 154.9035645, -382.4023438, 382.2908630
9: -171.8900909, 168.8322449, -171.0669861, 168.0516052, -339.9416809, 339.8992310

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8759073, upper bound: 338.8742086
time: 9.32 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8798811, upper bound: 338.8794363
time: 9.06 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8868318, upper bound: 338.8852913
time: 8.74 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -180.5018005, 143.5478363, -187.0249176, 148.7090454, -329.2108459, 330.5727539
1: -151.5481262, 127.5114594, -156.9591064, 132.0491638, -283.5972900, 284.4705811
2: -198.9959259, 128.9829102, -206.1374054, 133.6064758, -332.6024170, 335.1203003
3: -211.2698059, 111.6172409, -218.9130402, 115.6342392, -326.9039612, 330.5302734
4: -193.8441162, 148.1964874, -200.7862701, 153.4700317, -347.3141479, 348.9827271
5: -173.6514893, 134.9976196, -179.8272095, 139.7757721, -313.4272156, 314.8248291
6: -166.4092102, 160.2393341, -172.3595581, 165.9728241, -332.3819885, 332.5988770
7: -180.9326935, 152.4826202, -187.4101410, 157.8600922, -338.7927246, 339.8927612
8: -218.7379608, 149.5603638, -226.6953888, 154.9080811, -373.6460266, 376.2557373
9: -165.2308197, 162.3405914, -171.0719910, 168.0565186, -333.2872620, 333.4124756

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8813169, upper bound: 338.8822467
time: 8.52 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8850038, upper bound: 338.8835604
time: 9.06 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8668827, upper bound: 338.8684561
time: 9.90 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8864774, upper bound: 338.8861433
time: 8.49 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8744282, upper bound: 338.8752597
time: 9.15 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8889165, upper bound: 338.8876465
time: 9.89 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8931710, upper bound: 338.8910310
time: 9.11 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 87.83 seconds
IS_A1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 87.83
Output dim: 7, lower bound: -338.8839232, upper bound: 338.8820944
IS_A1_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 87.83
Output dim: 7, lower bound: -338.8802510, upper bound: 338.8788602
IS_A1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 87.83
Output dim: 7, lower bound: -338.8816923, upper bound: 338.8808662
IS_A1_B1_B2_A2, status: Status.VERIFIED, split count: 4, time: 87.83
Output dim: 7, lower bound: -338.8859994, upper bound: 338.8844409
IS_A1_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 87.83
Output dim: 7, lower bound: -338.8798811, upper bound: 338.8794363
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 87.83
Output dim: 7, lower bound: -338.8868318, upper bound: 338.8852913
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 87.83
Output dim: 7, lower bound: -338.8889165, upper bound: 338.8876465
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 87.83
Output dim: 7, lower bound: -338.8931710, upper bound: 338.8910310

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -185.0810242, 147.1473236, -187.0194550, 148.7047272, -333.7857666, 334.1667175
1: -155.3993225, 130.6436310, -156.9544983, 132.0453033, -287.4446411, 287.5980835
2: -203.9319153, 132.1785278, -206.1313629, 133.6025543, -337.5344849, 338.3098755
3: -216.5323334, 114.2830429, -218.9066010, 115.6308594, -332.1632080, 333.1896057
4: -198.7969360, 151.9068146, -200.7803497, 153.4655457, -352.2624817, 352.6871643
5: -178.0812225, 138.3795013, -179.8219147, 139.7716980, -317.8529053, 318.2014160
6: -170.6015167, 164.1981201, -172.3545074, 165.9679413, -336.5693970, 336.5526123
7: -185.4013214, 156.2402191, -187.4045868, 157.8554840, -343.2567444, 343.6448059
8: -224.1778870, 153.3186951, -226.6887207, 154.9035645, -379.0814209, 380.0073853
9: -169.3872223, 166.3511658, -171.0669861, 168.0516052, -337.4387817, 337.4181213

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8734759, upper bound: 338.8716254
time: 9.92 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8757322, upper bound: 338.8739343
time: 8.54 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8827495, upper bound: 338.8806770
time: 8.47 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8742774, upper bound: 338.8724619
time: 8.03 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -175.9069061, 139.9434967, -187.0249176, 148.7090454, -324.6159668, 326.9683533
1: -147.7058258, 124.2846375, -156.9591064, 132.0491638, -279.7549744, 281.2437134
2: -193.9730072, 125.7046127, -206.1374054, 133.6064758, -327.5794678, 331.8420105
3: -205.9022980, 108.7716980, -218.9130402, 115.6342392, -321.5364685, 327.6847229
4: -188.9238129, 144.3848419, -200.7862701, 153.4700317, -342.3938599, 345.1711121
5: -169.2644348, 131.5650635, -179.8272095, 139.7757721, -309.0401917, 311.3922729
6: -162.1583862, 156.2000885, -172.3595581, 165.9728241, -328.1311951, 328.5596313
7: -176.3467407, 148.6638031, -187.4101410, 157.8600922, -334.2068176, 336.0739136
8: -213.1959686, 145.7361755, -226.6953888, 154.9080811, -368.1040039, 372.4315796
9: -161.0660095, 158.2003326, -171.0719910, 168.0565186, -329.1225281, 329.2723389

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8725717, upper bound: 338.8730400
time: 9.34 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=341.21295166015625
rel_dist={7: [-338.9345586391237, 338.9345586391237]}

## Binary search (step 3) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9184436, upper bound: 338.9178927
time: 9.46 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9074701, upper bound: 338.9074701
time: 7.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.99 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.99
Output dim: 7, lower bound: -338.9184436, upper bound: 338.9178927
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.99
Output dim: 7, lower bound: -338.9074701, upper bound: 338.9074701

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -184.7211609, 146.8837128, -328.2977295, 328.9904175
1: -152.3174133, 128.1564331, -155.1049194, 130.5072479, -282.8246460, 283.2612915
2: -200.0067291, 129.6382141, -203.6430969, 131.9900055, -331.9967346, 333.2812805
3: -212.3461914, 112.1826477, -216.2613983, 114.2387161, -326.5848389, 328.4439697
4: -194.8282928, 148.9458313, -198.3916016, 151.6778870, -346.5061646, 347.3374329
5: -174.5253906, 135.6783600, -177.7157135, 138.1587830, -312.6841736, 313.3940430
6: -167.2519073, 161.0517578, -170.2996063, 163.9844208, -331.2363281, 331.3513794
7: -181.8564148, 153.2529755, -185.1827240, 156.0302124, -337.8866272, 338.4356995
8: -219.8477783, 150.3132477, -223.8251801, 153.0471344, -372.8948975, 374.1384277
9: -166.0647430, 163.1612549, -169.0843506, 166.1456757, -332.2103577, 332.2455139

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9105436, upper bound: 338.9108948
time: 9.59 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8935806, upper bound: 338.8921755
time: 8.43 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -180.5725403, 143.6135406, -184.1629181, 146.4421234, -327.0146484, 327.7764587
1: -151.4880676, 127.4342117, -154.6336212, 130.1104736, -281.5985413, 282.0678406
2: -199.0630951, 129.0347137, -203.0283051, 131.5933533, -330.6564331, 332.0630188
3: -211.2796173, 111.5729218, -215.6005402, 113.8926544, -325.1722717, 327.1734619
4: -193.8635254, 148.0949554, -197.7890930, 151.2166595, -345.0801392, 345.8840332
5: -173.5881195, 134.9150391, -177.1758423, 137.7401581, -311.3282166, 312.0908813
6: -166.5021667, 160.2713776, -169.7857056, 163.4888916, -329.9910583, 330.0570374
7: -180.9860382, 152.5078735, -184.6210022, 155.5603333, -336.5463562, 337.1288757
8: -218.9145355, 149.4556274, -223.1540375, 152.5859375, -371.5003662, 372.6096497
9: -165.2428284, 162.2864838, -168.5739594, 165.6410980, -330.8839111, 330.8604431

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8849713, upper bound: 338.8868561
time: 8.72 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8734921, upper bound: 338.8734921
time: 6.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.77 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.77
Output dim: 7, lower bound: -338.9105436, upper bound: 338.9108948
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.77
Output dim: 7, lower bound: -338.8935806, upper bound: 338.8921755
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.77
Output dim: 7, lower bound: -338.8849713, upper bound: 338.8868561
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 16.77
Output dim: 7, lower bound: -338.8734921, upper bound: 338.8734921

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -181.8114929, 144.5773010, -325.9913330, 326.0807495
1: -152.3174133, 128.1564331, -152.6432495, 128.4476624, -280.7650757, 280.7996216
2: -200.0067291, 129.6382141, -200.4200897, 129.9217987, -329.9285278, 330.0582886
3: -212.3461914, 112.1826477, -212.8362732, 112.4366455, -324.7827454, 325.0188599
4: -194.8282928, 148.9458313, -195.2288971, 149.2648773, -344.0931702, 344.1747437
5: -174.5253906, 135.6783600, -174.8884888, 135.9642639, -310.4896545, 310.5668335
6: -167.2519073, 161.0517578, -167.6027069, 161.3926849, -328.6445923, 328.6544800
7: -181.8564148, 153.2529755, -182.2293854, 153.5557556, -335.4121704, 335.4823608
8: -219.8477783, 150.3132477, -220.3479004, 150.6572723, -370.5050659, 370.6611328
9: -166.0647430, 163.1612549, -166.3887329, 163.5004578, -329.5650940, 329.5499573

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8975903, upper bound: 338.8995376
time: 8.97 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9004007, upper bound: 338.9010411
time: 8.45 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8858996, upper bound: 338.8847504
time: 10.14 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -181.3368530, 144.2081757, -187.0249176, 148.7090454, -330.0458984, 331.2330322
1: -152.2519684, 128.1017609, -156.9591064, 132.0491638, -284.3011475, 285.0608215
2: -199.9212799, 129.5831909, -206.1374054, 133.6064758, -333.5277710, 335.7205811
3: -212.2551575, 112.1349030, -218.9130402, 115.6342392, -327.8893127, 331.0479431
4: -194.7445526, 148.8819122, -200.7862701, 153.4700317, -348.2145996, 349.6681519
5: -174.4503784, 135.6201630, -179.8272095, 139.7757721, -314.2261047, 315.4473877
6: -167.1803741, 160.9831543, -172.3595581, 165.9728241, -333.1531982, 333.3427124
7: -181.7781677, 153.1874542, -187.4101410, 157.8600922, -339.6382141, 340.5975647
8: -219.7553406, 150.2501373, -226.6953888, 154.9080811, -374.6634216, 376.9454956
9: -165.9934692, 163.0912018, -171.0719910, 168.0565186, -334.0499573, 334.1631775

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8782852, upper bound: 338.8794129
time: 8.91 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8826852, upper bound: 338.8812942
time: 9.27 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8554957, upper bound: 338.8568556
time: 9.05 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8837201, upper bound: 338.8834294
time: 9.09 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8701309, upper bound: 338.8711041
time: 8.72 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8859136, upper bound: 338.8850237
time: 10.22 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8935806, upper bound: 338.8921755
time: 9.24 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -180.5725403, 143.6135406, -181.2551880, 144.1372070, -324.7097473, 324.8687134
1: -151.4880676, 127.4342117, -152.1736145, 128.0522766, -279.5403442, 279.6077881
2: -199.0630951, 129.0347137, -199.8074646, 129.5264893, -328.5895996, 328.8421631
3: -211.2796173, 111.5729218, -212.1777191, 112.0917358, -323.3713379, 323.7506104
4: -193.8635254, 148.0949554, -194.6284027, 148.8051910, -342.6687012, 342.7233276
5: -173.5881195, 134.9150391, -174.3505096, 135.5470886, -309.1351929, 309.2655640
6: -166.5021667, 160.2713776, -167.0904694, 160.8989258, -327.4010925, 327.3618469
7: -180.9860382, 152.5078735, -181.6695709, 153.0875092, -334.0735474, 334.1773682
8: -218.9145355, 149.4556274, -219.6791077, 150.1975403, -369.1120605, 369.1346741
9: -165.2428284, 162.2864838, -165.8801270, 162.9976349, -328.2404175, 328.1666260

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8595418, upper bound: 338.8593281
time: 9.11 seconds

## Relational analysis of IS_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8697198, upper bound: 338.8698743
time: 7.95 seconds

## Relational analysis of IS_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8714032, upper bound: 338.8726554
time: 7.39 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8826219, upper bound: 338.8849852
time: 8.25 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 58.89 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 58.89
Output dim: 7, lower bound: -338.9004007, upper bound: 338.9010411
IS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 58.89
Output dim: 7, lower bound: -338.8858996, upper bound: 338.8847504
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 58.89
Output dim: 7, lower bound: -338.8859136, upper bound: 338.8850237
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 58.89
Output dim: 7, lower bound: -338.8935806, upper bound: 338.8921755
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 58.89
Output dim: 7, lower bound: -338.8714032, upper bound: 338.8726554
IS_A2_B1_B2, status: Status.VERIFIED, split count: 3, time: 58.89
Output dim: 7, lower bound: -338.8826219, upper bound: 338.8849852

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -181.4140472, 144.2692413, -172.1764526, 136.9344330, -318.3484192, 316.4456787
1: -152.3174133, 128.1564331, -144.5820465, 121.6016312, -273.9190369, 272.7384644
2: -200.0067291, 129.6382141, -189.8091583, 123.1007156, -323.1074524, 319.4473877
3: -212.3461914, 112.1826477, -201.4711914, 106.4826202, -318.8287659, 313.6538391
4: -194.8282928, 148.9458313, -184.8327942, 141.3356018, -336.1638794, 333.7786255
5: -174.5253906, 135.6783600, -165.5833740, 128.7392578, -303.2646179, 301.2616577
6: -167.2519073, 161.0517578, -158.7264252, 152.8542480, -320.1061401, 319.7781982
7: -181.8564148, 153.2529755, -172.5192719, 145.4586487, -327.3150635, 325.7722168
8: -219.8477783, 150.3132477, -208.8358765, 142.6981049, -362.5458984, 359.1491089
9: -166.0647430, 163.1612549, -157.5827789, 154.7900085, -320.8547058, 320.7439575

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8881101, upper bound: 338.8896051
time: 11.22 seconds

## Relational analysis of IS_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8801682, upper bound: 338.8783452
time: 9.24 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8766620, upper bound: 338.8752455
time: 9.47 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -180.4244537, 143.4866333, -187.0249176, 148.7090454, -329.1334839, 330.5115356
1: -151.4825287, 127.4566956, -156.9591064, 132.0491638, -283.5316772, 284.4158020
2: -198.9103241, 128.9277954, -206.1374054, 133.6064758, -332.5167847, 335.0651855
3: -211.1785736, 111.5693970, -218.9130402, 115.6342392, -326.8127441, 330.4824219
4: -193.7601776, 148.1323853, -200.7862701, 153.4700317, -347.2301941, 348.9186401
5: -173.5763855, 134.9393158, -179.8272095, 139.7757721, -313.3521423, 314.7665405
6: -166.3375092, 160.1705780, -172.3595581, 165.9728241, -332.3103333, 332.5301514
7: -180.8542175, 152.4169464, -187.4101410, 157.8600922, -338.7142639, 339.8270874
8: -218.6452942, 149.4971466, -226.6953888, 154.9080811, -373.5533752, 376.1925049
9: -165.1594238, 162.2703705, -171.0719910, 168.0565186, -333.2159119, 333.3422852

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8782852, upper bound: 338.8794129
time: 9.36 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8826852, upper bound: 338.8812942
time: 9.96 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8554957, upper bound: 338.8568556
time: 9.23 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8837201, upper bound: 338.8834294
time: 8.79 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8701309, upper bound: 338.8711041
time: 8.86 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8866719, upper bound: 338.8856720
time: 9.46 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8914554, upper bound: 338.8895128
time: 8.25 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 87.45 seconds
IS_A1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 87.45
Output dim: 7, lower bound: -338.8801682, upper bound: 338.8783452
IS_A1_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 87.45
Output dim: 7, lower bound: -338.8766620, upper bound: 338.8752455
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 87.45
Output dim: 7, lower bound: -338.8866719, upper bound: 338.8856720
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 87.45
Output dim: 7, lower bound: -338.8914554, upper bound: 338.8895128

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -175.8298950, 139.8825531, -187.0249176, 148.7090454, -324.5389404, 326.9073792
1: -147.6405487, 124.2301102, -156.9591064, 132.0491638, -279.6896973, 281.1892090
2: -193.8877869, 125.6497650, -206.1374054, 133.6064758, -327.4942627, 331.7871704
3: -205.8115540, 108.7240753, -218.9130402, 115.6342392, -321.4457397, 327.6370544
4: -188.8402710, 144.3210907, -200.7862701, 153.4700317, -342.3103027, 345.1073303
5: -169.1896515, 131.5070038, -179.8272095, 139.7757721, -308.9653931, 311.3342285
6: -162.0870514, 156.1316223, -172.3595581, 165.9728241, -328.0598450, 328.4911804
7: -176.2686310, 148.5984497, -187.4101410, 157.8600922, -334.1286621, 336.0085449
8: -213.1037445, 145.6732178, -226.6953888, 154.9080811, -368.0117798, 372.3685913
9: -160.9948883, 158.1304474, -171.0719910, 168.0565186, -329.0513916, 329.2024536

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8672872, upper bound: 338.8676977
time: 8.15 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8754222, upper bound: 338.8744328
time: 8.63 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8645652, upper bound: 338.8642779
time: 8.90 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8765233, upper bound: 338.8757323
time: 8.57 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8745836, upper bound: 338.8737043
time: 8.58 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8746008, upper bound: 338.8736893
time: 9.41 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -177.6750336, 141.3350220, -187.0249176, 148.7090454, -326.3840942, 328.3599243
1: -149.1765442, 125.5289154, -156.9591064, 132.0491638, -281.2257080, 282.4880066
2: -195.9061127, 126.9706192, -206.1374054, 133.6064758, -329.5125732, 333.1080322
3: -207.9734039, 109.8746643, -218.9130402, 115.6342392, -323.6076355, 328.7876892
4: -190.8157806, 145.8615112, -200.7862701, 153.4700317, -344.2858276, 346.6477661
5: -170.9485931, 132.8877563, -179.8272095, 139.7757721, -310.7243652, 312.7149658
6: -163.8020020, 157.7540741, -172.3595581, 165.9728241, -329.7748413, 330.1136169
7: -178.1080322, 150.1296387, -187.4101410, 157.8600922, -335.9681091, 337.5397644
8: -215.3356476, 147.2202454, -226.6953888, 154.9080811, -370.2436829, 373.9156494
9: -162.6642151, 159.7973175, -171.0719910, 168.0565186, -330.7207336, 330.8692932

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8757826, upper bound: 338.8765166
time: 9.13 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8803408, upper bound: 338.8786695
time: 9.25 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8530323, upper bound: 338.8545353
time: 8.17 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8814638, upper bound: 338.8809274
time: 9.09 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8676845, upper bound: 338.8684996
time: 9.00 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8813847, upper bound: 338.8795700
time: 8.89 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8817368, upper bound: 338.8818296
time: 8.96 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8914554, upper bound: 338.8895128
time: 9.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 105.42 seconds
IS_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 105.42
Output dim: 7, lower bound: -338.8745836, upper bound: 338.8737043
IS_A1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 105.42
Output dim: 7, lower bound: -338.8746008, upper bound: 338.8736893
IS_A1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 105.42
Output dim: 7, lower bound: -338.8817368, upper bound: 338.8818296
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 105.42
Output dim: 7, lower bound: -338.8914554, upper bound: 338.8895128

## BFS IS instance: IS_A1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -177.2095947, 140.9642181, -187.0249176, 148.7090454, -325.9185791, 327.9890747
1: -148.7855072, 125.2016373, -156.9591064, 132.0491638, -280.8346558, 282.1607361
2: -195.3916626, 126.6387863, -206.1374054, 133.6064758, -328.9981384, 332.7761841
3: -207.4246063, 109.5878372, -218.9130402, 115.6342392, -323.0588379, 328.5008850
4: -190.3147430, 145.4797211, -200.7862701, 153.4700317, -343.7847900, 346.2659302
5: -170.4997711, 132.5407257, -179.8272095, 139.7757721, -310.2754822, 312.3679199
6: -163.3731232, 157.3411713, -172.3595581, 165.9728241, -329.3459473, 329.7007446
7: -177.6394196, 149.7363434, -187.4101410, 157.8600922, -335.4994812, 337.1464233
8: -214.7727661, 146.8386688, -226.6953888, 154.9080811, -369.6808167, 373.5340271
9: -162.2390289, 159.3791199, -171.0719910, 168.0565186, -330.2955017, 330.4511108

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8757826, upper bound: 338.8765166
time: 9.44 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8803408, upper bound: 338.8786695
time: 7.52 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8530323, upper bound: 338.8545353
time: 7.15 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 3): status=Status.UNKNOWN, k_low=10, k_high=10, k_mid=10, eps_mid=0.0390625, abs_max=341.21295166015625
rel_dist={7: [-338.9345347092476, 338.93453470122904]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.03515625
execution time: 1858.23 seconds
