## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 326.172941817
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905)
1: (-148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693)
2: (-195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773)
3: (-207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982)
4: (-189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597)
5: (-170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746)
6: (-163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143)
7: (-178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861)
8: (-213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505)
9: (-161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854)

## BASE Result
execution time: IAR + LP analysis = 1.22 + 11.01 = 12.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -326.2561776, upper bound: 326.2561776


# Binary Search by BASE starts (time budget: 2687.77 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=328.3682861328125
rel_dist={7: [-326.25613672106726, 326.2561367077651]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=328.3682861328125
rel_dist={7: [-326.2560128858547, 326.2560128858547]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=328.3682861328125
rel_dist={7: [-326.25584232239004, 326.2558422835341]}

## Binary Search Result
Binary search time: 44.52 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2643.25 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2081253, upper bound: 326.2135723
time: 9.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1916368, upper bound: 326.1916367
time: 9.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.55
Output dim: 7, lower bound: -326.2081253, upper bound: 326.2135723
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.55
Output dim: 7, lower bound: -326.1916368, upper bound: 326.1916367

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -176.1383057, 139.9498596, -176.8887177, 140.5452881, -316.6835632, 316.8385620
1: -148.1291351, 124.6177597, -148.7599487, 125.1486740, -273.2777710, 273.3777161
2: -194.3291779, 127.1356506, -195.1577606, 127.6752167, -322.0043945, 322.2933655
3: -206.5952301, 109.2217331, -207.4779510, 109.6864548, -316.2816772, 316.6996765
4: -188.8157654, 145.2543488, -189.6262207, 145.8749542, -334.6907349, 334.8805542
5: -169.4739380, 132.2519989, -170.1939697, 132.8175659, -302.2914734, 302.4459229
6: -162.5175018, 156.2819061, -163.2100983, 156.9458160, -319.4632874, 319.4920044
7: -177.6271973, 149.3514099, -178.3847504, 149.9835510, -327.6107178, 327.7361450
8: -212.9792480, 144.7171021, -213.8840027, 145.3365479, -358.3157654, 358.6010742
9: -161.1715698, 159.2395630, -161.8587646, 159.9163361, -321.0878906, 321.0983276

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1708131, upper bound: 326.1729793
time: 9.25 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2079489, upper bound: 326.2134803
time: 8.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -170.4180756, 135.3389435, -175.9783325, 139.8215179, -310.2395935, 311.3172607
1: -143.2613831, 120.4438782, -147.9944611, 124.5033569, -267.7647400, 268.4383545
2: -187.9421997, 122.8000488, -194.1520386, 127.0174637, -314.9596252, 316.9520569
3: -199.7481079, 105.6107330, -206.4063568, 109.1224136, -308.8705139, 312.0170593
4: -182.5223236, 140.3965607, -188.6426239, 145.1212311, -327.6435547, 329.0391846
5: -163.9718323, 127.7645874, -169.3196411, 132.1293640, -296.1011963, 297.0841980
6: -157.1690369, 151.1747589, -162.3692474, 156.1401978, -313.3092346, 313.5438843
7: -171.7068329, 144.4490356, -177.4651794, 149.2155304, -320.9223633, 321.9142151
8: -206.0433807, 139.8561096, -212.7858124, 144.5838623, -350.6272583, 352.6419067
9: -155.8437653, 153.9222107, -161.0245056, 159.0944977, -314.9382019, 314.9466248

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1732470, upper bound: 326.1734667
time: 7.01 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743237, upper bound: 326.1743237
time: 7.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.65 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 19.65
Output dim: 7, lower bound: -326.1708131, upper bound: 326.1729793
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 19.65
Output dim: 7, lower bound: -326.2079489, upper bound: 326.2134803
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.65
Output dim: 7, lower bound: -326.1732470, upper bound: 326.1734667
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.65
Output dim: 7, lower bound: -326.1743237, upper bound: 326.1743237

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -157.4274902, 125.0402985, -174.2997589, 138.4849091, -295.9123535, 299.3400269
1: -132.1860809, 111.2133713, -146.5617828, 123.2990952, -255.4851685, 257.7751465
2: -173.5393524, 113.3920746, -192.2865295, 125.7859039, -299.3252258, 305.6785889
3: -184.4552460, 97.5217896, -204.4180450, 108.0741653, -292.5294189, 301.9398193
4: -168.6831055, 129.6064301, -186.8412323, 143.7173157, -312.4003906, 316.4476624
5: -151.4199982, 118.0810623, -167.7002258, 130.8637390, -282.2837524, 285.7812805
6: -145.2441559, 139.6266327, -160.8203583, 154.6437378, -299.8878784, 300.4469910
7: -158.6709442, 133.3775024, -175.7675934, 147.7803802, -306.4512634, 309.1450806
8: -190.4136505, 129.1073303, -210.7613983, 143.1827698, -333.5963440, 339.8687134
9: -143.9047394, 142.2192688, -159.4739685, 157.5675201, -301.4722290, 301.6932373

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1492227, upper bound: 326.1517833
time: 7.91 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1481662, upper bound: 326.1506968
time: 7.73 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -173.5910034, 137.9245148, -176.8887177, 140.5452881, -314.1362610, 314.8132324
1: -145.9787598, 122.8077545, -148.7599487, 125.1486740, -271.1274109, 271.5676880
2: -191.5088501, 125.3013687, -195.1577606, 127.6752167, -319.1840210, 320.4591064
3: -203.5961456, 107.6344681, -207.4779510, 109.6864548, -313.2825623, 315.1124268
4: -186.0698700, 143.1406403, -189.6262207, 145.8749542, -331.9448242, 332.7668457
5: -167.0213928, 130.3332672, -170.1939697, 132.8175659, -299.8389587, 300.5271912
6: -160.1660004, 154.0150757, -163.2100983, 156.9458160, -317.1118164, 317.2251587
7: -175.0565033, 147.1963196, -178.3847504, 149.9835510, -325.0400391, 325.5810547
8: -209.9087219, 142.6050568, -213.8840027, 145.3365479, -355.2452698, 356.4890442
9: -158.8335114, 156.9300385, -161.8587646, 159.9163361, -318.7498169, 318.7888184

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1942942, upper bound: 326.1986206
time: 7.71 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1929198, upper bound: 326.2003238
time: 8.51 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1935772, upper bound: 326.2006471
time: 8.02 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -167.3458862, 132.9038391, -142.1247253, 112.9936829, -280.3395691, 275.0285645
1: -140.6753235, 118.2715759, -119.5217590, 100.5799255, -241.2552338, 237.7933350
2: -184.5508270, 120.6034698, -156.7992554, 102.8281403, -287.3789673, 277.4027100
3: -196.1401367, 103.7104111, -166.6612701, 88.1913376, -284.3314514, 270.3716125
4: -179.2141571, 137.8659058, -152.1929626, 117.2543488, -296.4685059, 290.0588684
5: -161.0087280, 125.4583206, -136.6900177, 106.7276077, -267.7363281, 262.1483459
6: -154.3451538, 148.4528809, -131.2433167, 126.1577072, -280.5028381, 279.6961975
7: -168.6161652, 141.8644104, -143.4209900, 120.7504272, -289.3665466, 285.2853699
8: -202.3460846, 137.3292084, -172.0432129, 116.7433701, -319.0894470, 309.3723450
9: -153.0466919, 151.1547089, -130.2050018, 128.6112213, -281.6578674, 281.3597107

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729265, upper bound: 326.1729265
time: 7.45 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729265, upper bound: 326.1734667
time: 6.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -170.4180756, 135.3389435, -161.5422058, 128.3623657, -298.7803955, 296.8811646
1: -143.2613831, 120.4438782, -135.8804932, 114.3282776, -257.5896606, 256.3243713
2: -187.9421997, 122.8000488, -178.2515411, 116.7346115, -304.6767578, 301.0515747
3: -199.7481079, 105.6107330, -189.4804993, 100.2120438, -299.9601440, 295.0911865
4: -182.5223236, 140.3965607, -173.1013336, 133.2491608, -315.7714844, 313.4978638
5: -163.9718323, 127.7645874, -155.4045563, 121.3122482, -285.2840881, 283.1690979
6: -157.1690369, 151.1747589, -149.0893860, 143.3796692, -300.5487061, 300.2640991
7: -171.7068329, 144.4490356, -162.9966125, 137.1132965, -308.8201294, 307.4456482
8: -206.0433807, 139.8561096, -195.4198303, 132.6754456, -338.7188110, 335.2759094
9: -155.8437653, 153.9222107, -147.8955383, 146.1271667, -301.9709473, 301.8177185

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1734667, upper bound: 326.1732470
time: 6.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1734667, upper bound: 326.1743237
time: 6.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.90 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 22.90
Output dim: 7, lower bound: -326.1492227, upper bound: 326.1517833
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 22.90
Output dim: 7, lower bound: -326.1481662, upper bound: 326.1506968
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 7, lower bound: -326.1929198, upper bound: 326.2003238
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 7, lower bound: -326.1935772, upper bound: 326.2006471
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 22.90
Output dim: 7, lower bound: -326.1729265, upper bound: 326.1729265
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 7, lower bound: -326.1729265, upper bound: 326.1734667
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 7, lower bound: -326.1734667, upper bound: 326.1732470
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 7, lower bound: -326.1734667, upper bound: 326.1743237

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -170.4152374, 135.4069519, -142.9952087, 113.6855774, -284.1008301, 278.4021301
1: -143.3056488, 120.5619888, -120.2532425, 101.1967163, -244.5023651, 240.8152161
2: -188.0025787, 123.0287094, -157.7607422, 103.4563980, -291.4589844, 280.7894287
3: -199.8675995, 105.6711121, -167.6863556, 88.7308502, -288.5984497, 273.3574829
4: -182.6492767, 140.5242767, -153.1332703, 117.9750061, -300.6242676, 293.6575317
5: -163.9598694, 127.9480438, -137.5261078, 107.3854294, -271.3453064, 265.4740906
6: -157.2458801, 151.2013397, -132.0471954, 126.9280167, -284.1738892, 283.2485352
7: -171.8602600, 144.5234528, -144.2999115, 121.4841537, -293.3444214, 288.8233337
8: -206.0849915, 139.9933472, -173.0928497, 117.4631882, -323.5481873, 313.0861816
9: -155.9412231, 154.0690308, -131.0026245, 129.3968506, -285.3380737, 285.0715942

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1902484, upper bound: 326.1968633
time: 8.97 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1902484, upper bound: 326.2003065
time: 8.57 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -173.5910034, 137.9245148, -162.4753876, 129.1040802, -302.6950684, 300.3999023
1: -145.9787598, 122.8077545, -136.6650543, 114.9896698, -260.9684143, 259.4728088
2: -191.5088501, 125.3013687, -179.2824554, 117.4086914, -308.9175110, 304.5838318
3: -203.5961456, 107.6344681, -190.5786285, 100.7902908, -304.3863525, 298.2131042
4: -186.0698700, 143.1406403, -174.1093292, 134.0215607, -320.0914307, 317.2499695
5: -167.0213928, 130.3332672, -156.3003693, 122.0177612, -289.0391541, 286.6336365
6: -160.1660004, 154.0150757, -149.9510803, 144.2054901, -304.3714905, 303.9661560
7: -175.0565033, 147.1963196, -163.9395142, 137.9003448, -312.9568481, 311.1358032
8: -209.9087219, 142.6050568, -196.5453186, 133.4469147, -343.3556213, 339.1503906
9: -158.8335114, 156.9300385, -148.7507019, 146.9695892, -305.8030701, 305.6807251

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1903770, upper bound: 326.1968768
time: 7.88 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1903770, upper bound: 326.2006471
time: 8.34 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -155.4737396, 123.4770508, -142.1247253, 112.9936829, -268.4673767, 265.6017456
1: -130.7179871, 109.9071198, -119.5217590, 100.5799255, -231.2978821, 229.4288635
2: -171.4756317, 112.1434097, -156.7992554, 102.8281403, -274.3037415, 268.9426575
3: -182.2270660, 96.3855667, -166.6612701, 88.1913376, -270.4183960, 263.0468140
4: -166.4269257, 128.1009521, -152.1929626, 117.2543488, -283.6812744, 280.2939148
5: -149.5692749, 116.5590286, -136.6900177, 106.7276077, -256.2968750, 253.2490540
6: -143.4204407, 137.9618683, -131.2433167, 126.1577072, -269.5780945, 269.2051392
7: -156.7140503, 131.9145813, -143.4209900, 120.7504272, -277.4644775, 275.3355408
8: -188.0634613, 127.5310440, -172.0432129, 116.7433701, -304.8068237, 299.5742493
9: -142.2478027, 140.4941254, -130.2050018, 128.6112213, -270.8589783, 270.6991272

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1600963, upper bound: 326.1574535
time: 6.09 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729318, upper bound: 326.1734667
time: 8.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -138.0074310, 109.6539154, -161.5422058, 128.3623657, -266.3697815, 271.1961060
1: -115.9966431, 97.5335312, -135.8804932, 114.3282776, -230.3249207, 233.4140320
2: -152.1828766, 99.6459579, -178.2515411, 116.7346115, -268.9174500, 277.8974915
3: -161.6824646, 85.5678101, -189.4804993, 100.2120438, -261.8944397, 275.0482788
4: -147.6199341, 113.7110596, -173.1013336, 133.2491608, -280.8690796, 286.8122864
5: -132.7144318, 103.4450531, -155.4045563, 121.3122482, -254.0266724, 258.8495789
6: -127.3721695, 122.4666061, -149.0893860, 143.3796692, -270.7518311, 271.5559998
7: -139.1227875, 117.1959381, -162.9966125, 137.1132965, -276.2360840, 280.1925049
8: -167.0466156, 113.1972351, -195.4198303, 132.6754456, -299.7220459, 308.6170654
9: -126.3391037, 124.7404633, -147.8955383, 146.1271667, -272.4662781, 272.6359558

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1574478, upper bound: 326.1600963
time: 6.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729265, upper bound: 326.1732470
time: 6.02 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -155.4737396, 123.4770508, -161.5422058, 128.3623657, -283.8360291, 285.0192261
1: -130.7179871, 109.9071198, -135.8804932, 114.3282776, -245.0462494, 245.7875977
2: -171.4756317, 112.1434097, -178.2515411, 116.7346115, -288.2101746, 290.3949585
3: -182.2270660, 96.3855667, -189.4804993, 100.2120438, -282.4390869, 285.8660583
4: -166.4269257, 128.1009521, -173.1013336, 133.2491608, -299.6760864, 301.2022705
5: -149.5692749, 116.5590286, -155.4045563, 121.3122482, -270.8815308, 271.9635010
6: -143.4204407, 137.9618683, -149.0893860, 143.3796692, -286.8001099, 287.0512390
7: -156.7140503, 131.9145813, -162.9966125, 137.1132965, -293.8273315, 294.9111328
8: -188.0634613, 127.5310440, -195.4198303, 132.6754456, -320.7388916, 322.9508667
9: -142.2478027, 140.4941254, -147.8955383, 146.1271667, -288.3749695, 288.3896484

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1574478, upper bound: 326.1608554
time: 6.31 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729265, upper bound: 326.1743237
time: 6.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 40.56 seconds
IS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 40.56
Output dim: 7, lower bound: -326.1902484, upper bound: 326.1968633
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 40.56
Output dim: 7, lower bound: -326.1902484, upper bound: 326.2003065
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 40.56
Output dim: 7, lower bound: -326.1903770, upper bound: 326.1968768
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 40.56
Output dim: 7, lower bound: -326.1903770, upper bound: 326.2006471
IS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 40.56
Output dim: 7, lower bound: -326.1600963, upper bound: 326.1574535
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 40.56
Output dim: 7, lower bound: -326.1729318, upper bound: 326.1734667
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 40.56
Output dim: 7, lower bound: -326.1574478, upper bound: 326.1600963
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 40.56
Output dim: 7, lower bound: -326.1729265, upper bound: 326.1732470
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 40.56
Output dim: 7, lower bound: -326.1574478, upper bound: 326.1608554
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 40.56
Output dim: 7, lower bound: -326.1729265, upper bound: 326.1743237

## BFS IS instance: IS_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -139.6036835, 110.9908524, -142.9952087, 113.6855774, -253.2892609, 253.9860535
1: -117.3924713, 98.7882767, -120.2532425, 101.1967163, -218.5891724, 219.0415192
2: -154.0077972, 101.0138550, -157.7607422, 103.4563980, -257.4641724, 258.7745667
3: -163.6932068, 86.6213455, -167.6863556, 88.7308502, -252.4240570, 254.3077087
4: -149.4756622, 115.1602554, -153.1332703, 117.9750061, -267.4506836, 268.2935181
5: -134.2642822, 104.8304825, -137.5261078, 107.3854294, -241.6497192, 242.3565979
6: -128.9173126, 123.9152451, -132.0471954, 126.9280167, -255.8453369, 255.9624329
7: -140.8767242, 118.6177063, -144.2999115, 121.4841537, -262.3608704, 262.9176025
8: -169.0063629, 114.6539764, -173.0928497, 117.4631882, -286.4695435, 287.7468262
9: -127.8898315, 126.3255463, -131.0026245, 129.3968506, -257.2866821, 257.3281860

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1703559, upper bound: 326.1750003
time: 8.23 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1537511, upper bound: 326.1579157
time: 8.56 seconds

## Relational analysis of IS_A1_A2_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1905017, upper bound: 326.1971742
time: 7.82 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -159.2537537, 126.5451431, -142.9952087, 113.6855774, -272.9393311, 269.5403442
1: -133.9487762, 112.7040100, -120.2532425, 101.1967163, -235.1454773, 232.9572296
2: -175.7185974, 115.0903320, -157.7607422, 103.4563980, -279.1749878, 272.8510437
3: -186.7866516, 98.7872543, -167.6863556, 88.7308502, -275.5174866, 266.4736023
4: -170.6362000, 131.3500977, -153.1332703, 117.9750061, -288.6112061, 284.4833679
5: -153.2019196, 119.5911255, -137.5261078, 107.3854294, -260.5873108, 257.1171875
6: -146.9773407, 141.3444824, -132.0471954, 126.9280167, -273.9053650, 273.3916626
7: -160.6888885, 135.1783905, -144.2999115, 121.4841537, -282.1730347, 279.4783020
8: -192.6630402, 130.7795715, -173.0928497, 117.4631882, -310.1261902, 303.8724365
9: -145.7953949, 144.0541077, -131.0026245, 129.3968506, -275.1922302, 275.0567322

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1545159, upper bound: 326.1593489
time: 8.92 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1862010, upper bound: 326.1964611
time: 9.27 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -139.6036835, 110.9908524, -162.4753876, 129.1040802, -268.7077637, 273.4662476
1: -117.3924713, 98.7882767, -136.6650543, 114.9896698, -232.3820953, 235.4533386
2: -154.0077972, 101.0138550, -179.2824554, 117.4086914, -271.4164429, 280.2963257
3: -163.6932068, 86.6213455, -190.5786285, 100.7902908, -264.4834595, 277.1999512
4: -149.4756622, 115.1602554, -174.1093292, 134.0215607, -283.4972229, 289.2695618
5: -134.2642822, 104.8304825, -156.3003693, 122.0177612, -256.2820435, 261.1308594
6: -128.9173126, 123.9152451, -149.9510803, 144.2054901, -273.1227417, 273.8663330
7: -140.8767242, 118.6177063, -163.9395142, 137.9003448, -278.7770691, 282.5572205
8: -169.0063629, 114.6539764, -196.5453186, 133.4469147, -302.4532471, 311.1992798
9: -127.8898315, 126.3255463, -148.7507019, 146.9695892, -274.8594360, 275.0762329

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1694074, upper bound: 326.1738982
time: 8.24 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1323829, upper bound: 326.1328577
time: 8.04 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1745031, upper bound: 326.1830714
time: 7.82 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1902484, upper bound: 326.1968768
time: 8.55 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -159.2537537, 126.5451431, -162.4753876, 129.1040802, -288.3578491, 289.0205383
1: -133.9487762, 112.7040100, -136.6650543, 114.9896698, -248.9384003, 249.3690643
2: -175.7185974, 115.0903320, -179.2824554, 117.4086914, -293.1272888, 294.3728027
3: -186.7866516, 98.7872543, -190.5786285, 100.7902908, -287.5769348, 289.3658142
4: -170.6362000, 131.3500977, -174.1093292, 134.0215607, -304.6577759, 305.4594116
5: -153.2019196, 119.5911255, -156.3003693, 122.0177612, -275.2196655, 275.8914795
6: -146.9773407, 141.3444824, -149.9510803, 144.2054901, -291.1828308, 291.2955627
7: -160.6888885, 135.1783905, -163.9395142, 137.9003448, -298.5892334, 299.1179199
8: -192.6630402, 130.7795715, -196.5453186, 133.4469147, -326.1098328, 327.3248901
9: -145.7953949, 144.0541077, -148.7507019, 146.9695892, -292.7649231, 292.8048096

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1694074, upper bound: 326.1838130
time: 8.68 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1364108, upper bound: 326.1461579
time: 9.05 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1323829, upper bound: 326.1656756
time: 8.42 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1540391, upper bound: 326.1595165
time: 8.65 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1859248, upper bound: 326.1967813
time: 8.47 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -154.0471344, 122.3498154, -142.1247253, 112.9936829, -267.0408325, 264.4745483
1: -129.5171356, 108.9033508, -119.5217590, 100.5799255, -230.0970459, 228.4251099
2: -169.9049835, 111.1283417, -156.7992554, 102.8281403, -272.7331238, 267.9275818
3: -180.5433960, 95.5057297, -166.6612701, 88.1913376, -268.7347412, 262.1669617
4: -164.8926544, 126.9265137, -152.1929626, 117.2543488, -282.1470032, 279.1194763
5: -148.2022095, 115.4997025, -136.6900177, 106.7276077, -254.9297943, 252.1897278
6: -142.1081848, 136.6978149, -131.2433167, 126.1577072, -268.2658691, 267.9411316
7: -155.2800598, 130.7175293, -143.4209900, 120.7504272, -276.0304260, 274.1385193
8: -186.3412781, 126.3616562, -172.0432129, 116.7433701, -303.0846558, 298.4048157
9: -140.9491119, 139.2110901, -130.2050018, 128.6112213, -269.5603027, 269.4160767

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1591139, upper bound: 326.1619234
time: 6.38 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1591139, upper bound: 326.1734667
time: 8.07 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -138.0074310, 109.6539154, -160.2100067, 127.3097458, -265.3171692, 269.8639221
1: -115.9966431, 97.5335312, -134.7605896, 113.3918610, -229.3885040, 232.2941284
2: -152.1828766, 99.6459579, -176.7845154, 115.7877426, -267.9705811, 276.4304810
3: -161.6824646, 85.5678101, -187.9063416, 99.3897858, -261.0722046, 273.4741516
4: -147.6199341, 113.7110596, -171.6714172, 132.1524963, -279.7724304, 285.3823853
5: -132.7144318, 103.4450531, -154.1284180, 120.3218460, -253.0362854, 257.5734863
6: -127.3721695, 122.4666061, -147.8616028, 142.2002106, -269.5723572, 270.3282166
7: -139.1227875, 117.1959381, -161.6565247, 135.9944458, -275.1172485, 278.8524475
8: -167.0466156, 113.1972351, -193.8110199, 131.5865173, -298.6331177, 307.0082092
9: -126.3391037, 124.7404633, -146.6811676, 144.9275055, -271.2666016, 271.4216309

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 65

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1734468, upper bound: 326.1731885
time: 6.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1733773, upper bound: 326.1731703
time: 8.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -155.4737396, 123.4770508, -160.2100067, 127.3097458, -282.7834778, 283.6870117
1: -130.7179871, 109.9071198, -134.7605896, 113.3918610, -244.1098480, 244.6676941
2: -171.4756317, 112.1434097, -176.7845154, 115.7877426, -287.2633362, 288.9279175
3: -182.2270660, 96.3855667, -187.9063416, 99.3897858, -281.6168518, 284.2919006
4: -166.4269257, 128.1009521, -171.6714172, 132.1524963, -298.5794067, 299.7723694
5: -149.5692749, 116.5590286, -154.1284180, 120.3218460, -269.8911133, 270.6874390
6: -143.4204407, 137.9618683, -147.8616028, 142.2002106, -285.6206055, 285.8234253
7: -156.7140503, 131.9145813, -161.6565247, 135.9944458, -292.7084961, 293.5710754
8: -188.0634613, 127.5310440, -193.8110199, 131.5865173, -319.6499634, 321.3420105
9: -142.2478027, 140.4941254, -146.6811676, 144.9275055, -287.1752625, 287.1752930

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=328.3682861328125
rel_dist={7: [-326.25613672106726, 326.2561367077651]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2010808, upper bound: 326.2040576
time: 9.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1915983, upper bound: 326.1915983
time: 5.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.20 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.20
Output dim: 7, lower bound: -326.2010808, upper bound: 326.2040576
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.20
Output dim: 7, lower bound: -326.1915983, upper bound: 326.1915983

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -176.1383057, 139.9498596, -176.8887177, 140.5452881, -316.6835632, 316.8385620
1: -148.1291351, 124.6177597, -148.7599487, 125.1486740, -273.2777710, 273.3777161
2: -194.3291779, 127.1356506, -195.1577606, 127.6752167, -322.0043945, 322.2933655
3: -206.5952301, 109.2217331, -207.4779510, 109.6864548, -316.2816772, 316.6996765
4: -188.8157654, 145.2543488, -189.6262207, 145.8749542, -334.6907349, 334.8805542
5: -169.4739380, 132.2519989, -170.1939697, 132.8175659, -302.2914734, 302.4459229
6: -162.5175018, 156.2819061, -163.2100983, 156.9458160, -319.4632874, 319.4920044
7: -177.6271973, 149.3514099, -178.3847504, 149.9835510, -327.6107178, 327.7361450
8: -212.9792480, 144.7171021, -213.8840027, 145.3365479, -358.3157654, 358.6010742
9: -161.1715698, 159.2395630, -161.8587646, 159.9163361, -321.0878906, 321.0983276

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1846324, upper bound: 326.1886966
time: 10.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1855032, upper bound: 326.1892913
time: 9.24 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -170.4180756, 135.3389435, -174.4445953, 138.6024780, -309.0205688, 309.7835388
1: -143.2613831, 120.4438782, -146.7050018, 123.4162979, -266.6776733, 267.1488342
2: -187.9421997, 122.8000488, -192.4578857, 125.9095001, -313.8516846, 315.2579346
3: -199.7481079, 105.6107330, -204.6009216, 108.1720123, -307.9200745, 310.2116699
4: -182.5223236, 140.3965607, -186.9858246, 143.8514404, -326.3737793, 327.3823853
5: -163.9718323, 127.7645874, -167.8468323, 130.9700470, -294.9418945, 295.6113892
6: -157.1690369, 151.1747589, -160.9527435, 154.7829895, -311.9520264, 312.1274414
7: -171.7068329, 144.4490356, -175.9159546, 147.9218597, -319.6286926, 320.3649902
8: -206.0433807, 139.8561096, -210.9357758, 143.3158569, -349.3592529, 350.7918701
9: -155.8437653, 153.9222107, -159.6192169, 157.7098694, -313.5536499, 313.5414124

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1730765, upper bound: 326.1732440
time: 7.13 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1742997, upper bound: 326.1742997
time: 8.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.74 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.74
Output dim: 7, lower bound: -326.1846324, upper bound: 326.1886966
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.74
Output dim: 7, lower bound: -326.1855032, upper bound: 326.1892913
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.74
Output dim: 7, lower bound: -326.1730765, upper bound: 326.1732440
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.74
Output dim: 7, lower bound: -326.1742997, upper bound: 326.1742997

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -162.7236481, 129.3147888, -142.9952087, 113.6855774, -276.4092407, 272.3099670
1: -136.8404541, 115.1329803, -120.2532425, 101.1967163, -238.0371704, 235.3862000
2: -179.5204315, 117.5400696, -157.7607422, 103.4563980, -282.9768066, 275.3007812
3: -190.8446350, 100.9247894, -167.6863556, 88.7308502, -279.5754395, 268.6111450
4: -174.3681488, 134.2051849, -153.1332703, 117.9750061, -292.3431396, 287.3384399
5: -156.5407867, 122.1786499, -137.5261078, 107.3854294, -263.9262085, 259.7047119
6: -150.1829376, 144.3982544, -132.0471954, 126.9280167, -277.1109619, 276.4454346
7: -164.1286926, 138.0644073, -144.2999115, 121.4841537, -285.6128235, 282.3643188
8: -196.8278503, 133.6828918, -173.0928497, 117.4631882, -314.2910461, 306.7757568
9: -148.9558258, 147.1561737, -131.0026245, 129.3968506, -278.3526611, 278.1587524

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1830086, upper bound: 326.1866934
time: 11.69 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1830086, upper bound: 326.1886966
time: 8.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -171.2346191, 136.0593414, -162.4753876, 129.1040802, -300.3386841, 298.5346985
1: -144.0141907, 121.1607590, -136.6650543, 114.9896698, -259.0038147, 257.8258057
2: -188.9277039, 123.6418686, -179.2824554, 117.4086914, -306.3363342, 302.9243164
3: -200.8455658, 106.1940536, -190.5786285, 100.7902908, -301.6357727, 296.7726746
4: -183.5368500, 141.2223969, -174.1093292, 134.0215607, -317.5583801, 315.3316956
5: -164.7480316, 128.5768585, -156.3003693, 122.0177612, -286.7657471, 284.8772278
6: -158.0068817, 151.9466705, -149.9510803, 144.2054901, -302.2123718, 301.8977661
7: -172.7104034, 145.2402344, -163.9395142, 137.9003448, -310.6107483, 309.1797485
8: -207.0807190, 140.6739349, -196.5453186, 133.4469147, -340.5275574, 337.2192383
9: -156.7121124, 154.8348083, -148.7507019, 146.9695892, -303.6817017, 303.5854797

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1830441, upper bound: 326.1866934
time: 8.38 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1830441, upper bound: 326.1892913
time: 8.88 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -157.4226074, 125.0387650, -140.6601562, 111.8295975, -269.2521667, 265.6989136
1: -132.3227844, 111.2553940, -118.2908020, 99.5421753, -231.8649597, 229.5461884
2: -173.5985260, 113.5090103, -155.1816864, 101.7711945, -275.3697205, 268.6907043
3: -184.4869995, 97.5731583, -164.9365234, 87.2835770, -271.7705688, 262.5096436
4: -168.5281372, 129.6931763, -150.6106873, 116.0418854, -284.5700073, 280.3038025
5: -151.4387970, 118.0092239, -135.2832642, 105.6207428, -257.0595398, 253.2924805
6: -145.2235565, 139.6617279, -129.8906860, 124.8616409, -270.0851440, 269.5524292
7: -158.6343842, 133.5161896, -141.9424591, 119.5158005, -278.1501770, 275.4586182
8: -190.4044189, 129.1679535, -170.2771454, 115.5321503, -305.9365845, 299.4450073
9: -144.0122681, 142.2161865, -128.8631287, 127.2893295, -271.3016052, 271.0792542

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1728975, upper bound: 326.1728975
time: 6.99 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1728975, upper bound: 326.1732440
time: 7.48 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -165.3367157, 131.3066406, -159.9693909, 127.1122742, -292.4489746, 291.2760315
1: -138.9970398, 116.8610916, -134.5582733, 113.2135010, -252.2105408, 251.4193573
2: -182.3429108, 119.1748962, -176.5139923, 115.5987091, -297.9415894, 295.6888428
3: -193.7922821, 102.4741821, -187.6297607, 99.2374496, -293.0296631, 290.1039124
4: -177.0479126, 136.2158966, -171.4023590, 131.9472656, -308.9951477, 307.6181641
5: -159.0753174, 123.9535217, -153.8945770, 120.1230774, -279.1983948, 277.8480835
6: -152.4939575, 146.6817780, -147.6371307, 141.9877472, -294.4816895, 294.3188782
7: -166.6072388, 140.1860046, -161.4076233, 135.7868500, -302.3941040, 301.5936279
8: -199.9288635, 135.6666412, -193.5231323, 131.3751373, -331.3040161, 329.1897583
9: -151.2205963, 149.3556976, -146.4542694, 144.7074432, -295.9280090, 295.8099670

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1732440, upper bound: 326.1730765
time: 7.15 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1732440, upper bound: 326.1742997
time: 7.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.17 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 7, lower bound: -326.1830086, upper bound: 326.1866934
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 7, lower bound: -326.1830086, upper bound: 326.1886966
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 7, lower bound: -326.1830441, upper bound: 326.1866934
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 7, lower bound: -326.1830441, upper bound: 326.1892913
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 16.17
Output dim: 7, lower bound: -326.1728975, upper bound: 326.1728975
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 7, lower bound: -326.1728975, upper bound: 326.1732440
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 7, lower bound: -326.1732440, upper bound: 326.1730765
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 7, lower bound: -326.1732440, upper bound: 326.1742997

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -142.2155457, 113.0665283, -142.9952087, 113.6855774, -255.9011230, 256.0616760
1: -119.5971680, 100.6445160, -120.2532425, 101.1967163, -220.7938690, 220.8977356
2: -156.8995361, 102.8950119, -157.7607422, 103.4563980, -260.3559265, 260.6557312
3: -166.7689209, 88.2479248, -167.6863556, 88.7308502, -255.4997711, 255.9342804
4: -152.2905579, 117.3294754, -153.1332703, 117.9750061, -270.2655334, 270.4627380
5: -136.7775879, 106.7972031, -137.5261078, 107.3854294, -244.1630249, 244.3233032
6: -131.3273315, 126.2379913, -132.0471954, 126.9280167, -258.2553406, 258.2851868
7: -143.5122375, 120.8269882, -144.2999115, 121.4841537, -264.9963989, 265.1268921
8: -172.1525116, 116.8194199, -173.0928497, 117.4631882, -289.6156921, 289.9122620
9: -130.2881470, 128.6931000, -131.0026245, 129.3968506, -259.6849976, 259.6956482

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1671741, upper bound: 326.1719194
time: 8.37 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1832455, upper bound: 326.1871089
time: 8.60 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -161.7462463, 128.5256042, -142.9952087, 113.6855774, -275.4317932, 271.5207520
1: -136.0521088, 114.4738770, -120.2532425, 101.1967163, -237.2488251, 234.7270813
2: -178.4775238, 116.8844299, -157.7607422, 103.4563980, -281.9339294, 274.6451721
3: -189.7207489, 100.3388824, -167.6863556, 88.7308502, -278.4515686, 268.0252380
4: -173.3217468, 133.4183502, -153.1332703, 117.9750061, -291.2967529, 286.5516357
5: -155.6003876, 121.4683685, -137.5261078, 107.3854294, -262.9858093, 258.9944763
6: -149.2780762, 143.5605011, -132.0471954, 126.9280167, -276.2060852, 275.6076660
7: -163.2038574, 137.2861786, -144.2999115, 121.4841537, -284.6879883, 281.5860901
8: -195.6663208, 132.8450928, -173.0928497, 117.4631882, -313.1295166, 305.9379272
9: -148.0830536, 146.3120728, -131.0026245, 129.3968506, -277.4799194, 277.3146667

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1747495, upper bound: 326.1802217
time: 8.00 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1753603, upper bound: 326.1813699
time: 9.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -142.2155457, 113.0665283, -162.4753876, 129.1040802, -271.3196411, 275.5418701
1: -119.5971680, 100.6445160, -136.6650543, 114.9896698, -234.5867920, 237.3095703
2: -156.8995361, 102.8950119, -179.2824554, 117.4086914, -274.3081970, 282.1774597
3: -166.7689209, 88.2479248, -190.5786285, 100.7902908, -267.5591431, 278.8265381
4: -152.2905579, 117.3294754, -174.1093292, 134.0215607, -286.3120728, 291.4387817
5: -136.7775879, 106.7972031, -156.3003693, 122.0177612, -258.7953491, 263.0975647
6: -131.3273315, 126.2379913, -149.9510803, 144.2054901, -275.5328369, 276.1890869
7: -143.5122375, 120.8269882, -163.9395142, 137.9003448, -281.4125977, 284.7665100
8: -172.1525116, 116.8194199, -196.5453186, 133.4469147, -305.5994263, 313.3647461
9: -130.2881470, 128.6931000, -148.7507019, 146.9695892, -277.2577515, 277.4437561

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1357740, upper bound: 326.1377229
time: 10.23 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1724453, upper bound: 326.1761376
time: 9.64 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1751921, upper bound: 326.1787080
time: 9.62 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -161.7462463, 128.5256042, -162.4753876, 129.1040802, -290.8503418, 291.0009460
1: -136.0521088, 114.4738770, -136.6650543, 114.9896698, -251.0417480, 251.1389160
2: -178.4775238, 116.8844299, -179.2824554, 117.4086914, -295.8862000, 296.1668701
3: -189.7207489, 100.3388824, -190.5786285, 100.7902908, -290.5109863, 290.9174805
4: -173.3217468, 133.4183502, -174.1093292, 134.0215607, -307.3432922, 307.5276794
5: -155.6003876, 121.4683685, -156.3003693, 122.0177612, -277.6181641, 277.7687378
6: -149.2780762, 143.5605011, -149.9510803, 144.2054901, -293.4835815, 293.5115662
7: -163.2038574, 137.2861786, -163.9395142, 137.9003448, -301.1041870, 301.2256775
8: -195.6663208, 132.8450928, -196.5453186, 133.4469147, -329.1131897, 329.3904114
9: -148.0830536, 146.3120728, -148.7507019, 146.9695892, -295.0526428, 295.0627747

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1357740, upper bound: 326.1493735
time: 8.06 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1650344, upper bound: 326.1723839
time: 9.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1830086, upper bound: 326.1892913
time: 9.45 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -155.4737396, 123.4770508, -140.6601562, 111.8295975, -267.3032837, 264.1371460
1: -130.7179871, 109.9071198, -118.2908020, 99.5421753, -230.2601624, 228.1978912
2: -171.4756317, 112.1434097, -155.1816864, 101.7711945, -273.2467957, 267.3251038
3: -182.2270660, 96.3855667, -164.9365234, 87.2835770, -269.5106506, 261.3220520
4: -166.4269257, 128.1009521, -150.6106873, 116.0418854, -282.4688110, 278.7116394
5: -149.5692749, 116.5590286, -135.2832642, 105.6207428, -255.1900024, 251.8422852
6: -143.4204407, 137.9618683, -129.8906860, 124.8616409, -268.2820129, 267.8525391
7: -156.7140503, 131.9145813, -141.9424591, 119.5158005, -276.2298584, 273.8569641
8: -188.0634613, 127.5310440, -170.2771454, 115.5321503, -303.5956116, 297.8081360
9: -142.2478027, 140.4941254, -128.8631287, 127.2893295, -269.5370789, 269.3572083

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1569993, upper bound: 326.1554648
time: 8.10 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729003, upper bound: 326.1732440
time: 6.20 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -138.0074310, 109.6539154, -159.9693909, 127.1122742, -265.1196899, 269.6232910
1: -115.9966431, 97.5335312, -134.5582733, 113.2135010, -229.2101440, 232.0917969
2: -152.1828766, 99.6459579, -176.5139923, 115.5987091, -267.7815552, 276.1599426
3: -161.6824646, 85.5678101, -187.6297607, 99.2374496, -260.9198608, 273.1975708
4: -147.6199341, 113.7110596, -171.4023590, 131.9472656, -279.5671997, 285.1133118
5: -132.7144318, 103.4450531, -153.8945770, 120.1230774, -252.8374939, 257.3396301
6: -127.3721695, 122.4666061, -147.6371307, 141.9877472, -269.3599243, 270.1037292
7: -139.1227875, 117.1959381, -161.4076233, 135.7868500, -274.9096375, 278.6035461
8: -167.0466156, 113.1972351, -193.5231323, 131.3751373, -298.4217529, 306.7203674
9: -126.3391037, 124.7404633, -146.4542694, 144.7074432, -271.0465088, 271.1947327

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 65

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1622463, upper bound: 326.1626613
time: 8.06 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1644628, upper bound: 326.1646674
time: 9.99 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -155.4737396, 123.4770508, -159.9693909, 127.1122742, -282.5859680, 283.4463501
1: -130.7179871, 109.9071198, -134.5582733, 113.2135010, -243.9314728, 244.4653778
2: -171.4756317, 112.1434097, -176.5139923, 115.5987091, -287.0742798, 288.6573792
3: -182.2270660, 96.3855667, -187.6297607, 99.2374496, -281.4645081, 284.0153198
4: -166.4269257, 128.1009521, -171.4023590, 131.9472656, -298.3741760, 299.5032959
5: -149.5692749, 116.5590286, -153.8945770, 120.1230774, -269.6923218, 270.4536133
6: -143.4204407, 137.9618683, -147.6371307, 141.9877472, -285.4081726, 285.5989685
7: -156.7140503, 131.9145813, -161.4076233, 135.7868500, -292.5009155, 293.3222046
8: -188.0634613, 127.5310440, -193.5231323, 131.3751373, -319.4385986, 321.0541687
9: -142.2478027, 140.4941254, -146.4542694, 144.7074432, -286.9551697, 286.9483948

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1554615, upper bound: 326.1574385
time: 7.07 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1728975, upper bound: 326.1742997
time: 6.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 45.56 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1671741, upper bound: 326.1719194
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1832455, upper bound: 326.1871089
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1747495, upper bound: 326.1802217
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1753603, upper bound: 326.1813699
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1724453, upper bound: 326.1761376
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1751921, upper bound: 326.1787080
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1650344, upper bound: 326.1723839
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1830086, upper bound: 326.1892913
IS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1569993, upper bound: 326.1554648
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1729003, upper bound: 326.1732440
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1622463, upper bound: 326.1626613
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1644628, upper bound: 326.1646674
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1554615, upper bound: 326.1574385
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 45.56
Output dim: 7, lower bound: -326.1728975, upper bound: 326.1742997

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -141.8993835, 112.8166962, -141.6808014, 112.6469727, -254.5463562, 254.4974823
1: -119.3313980, 100.4222412, -119.1483002, 100.2726593, -219.6040497, 219.5705414
2: -156.5513306, 102.6702194, -156.3132324, 102.5220642, -259.0733337, 258.9834290
3: -166.3953094, 88.0527725, -166.1329956, 87.9193649, -254.3146667, 254.1857605
4: -151.9511871, 117.0690536, -151.7224579, 116.8925018, -268.8436584, 268.7915039
5: -136.4747467, 106.5621338, -136.2671356, 106.4081345, -242.8828735, 242.8292694
6: -131.0359497, 125.9580612, -130.8355865, 125.7642670, -256.8002319, 256.7936401
7: -143.1941223, 120.5615158, -142.9773254, 120.3805771, -263.5747070, 263.5387268
8: -171.7707214, 116.5609283, -171.5053864, 116.3885574, -288.1592712, 288.0663147
9: -129.9998779, 128.4083862, -129.8041077, 128.2133026, -258.2131653, 258.2124939

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1778657, upper bound: 326.1805101
time: 8.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1778657, upper bound: 326.1871089
time: 8.60 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -143.3569946, 113.9693680, -137.1368103, 109.0422974, -252.3992767, 251.1061707
1: -120.5040817, 101.4462357, -115.2905807, 97.0360489, -217.5401306, 216.7367554
2: -158.1473083, 103.7059708, -151.2754059, 99.2400818, -257.3873291, 254.9813690
3: -168.0892334, 88.9266586, -160.7851868, 85.0791855, -253.1684265, 249.7118073
4: -153.5719604, 118.2216263, -146.8407593, 113.1213303, -266.6932983, 265.0623779
5: -137.8731384, 107.7728729, -131.8671875, 103.0176697, -240.8908081, 239.6400299
6: -132.3435974, 127.2484055, -126.6511841, 121.7283707, -254.0719604, 253.8995972
7: -144.7031708, 121.8023148, -138.3971710, 116.5420227, -261.2451477, 260.1994934
8: -173.4611206, 117.6059647, -166.0185394, 112.5877304, -286.0488586, 283.6245117
9: -131.3080139, 129.7119904, -125.6520004, 124.0998993, -255.4079132, 255.3639832

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1699950, upper bound: 326.1728023
time: 9.11 seconds

## Relational analysis of IS_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764140, upper bound: 326.1802217
time: 9.37 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -148.8334198, 118.2977142, -137.9898224, 109.7234726, -258.5568542, 256.2874451
1: -125.1796722, 105.3616180, -116.0402451, 97.6640549, -222.8437195, 221.4018402
2: -164.2277985, 107.6696091, -152.2362061, 99.8866577, -264.1144409, 259.9058228
3: -174.5748596, 92.3330460, -161.8153229, 85.6275711, -260.2024231, 254.1483307
4: -159.4275665, 122.7517853, -147.7479858, 113.8389587, -273.2665405, 270.4997559
5: -143.1475372, 111.8415833, -132.7001801, 103.6584702, -246.8059998, 244.5417633
6: -137.3847961, 132.1217651, -127.4376297, 122.4952393, -259.8800354, 259.5593872
7: -150.2423859, 126.4616089, -139.2761993, 117.2908859, -267.5332642, 265.7377930
8: -180.0863342, 122.1122360, -167.0536346, 113.3008575, -293.3872070, 289.1658020
9: -136.3240051, 134.6579590, -126.4445724, 124.8792038, -261.2031860, 261.1025391

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1744239, upper bound: 326.1787699
time: 9.75 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1744239, upper bound: 326.1813699
time: 9.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -136.3581543, 108.4240952, -144.0738983, 114.5385132, -250.8966675, 252.4979858
1: -114.6355820, 96.4849243, -121.1063843, 101.9535370, -216.5890808, 217.5913086
2: -150.4155579, 98.6794205, -158.9391174, 104.2212906, -254.6368408, 257.6184692
3: -159.8690643, 84.5970459, -168.9327545, 89.3707581, -249.2398071, 253.5297852
4: -145.9993439, 112.4766769, -154.3462524, 118.8146210, -264.8139343, 266.8229370
5: -131.1196442, 102.4302979, -138.5614929, 108.3134995, -239.4331360, 240.9917755
6: -125.9322891, 121.0392990, -133.0054626, 127.8825150, -253.8148041, 254.0447693
7: -137.6108398, 115.8859940, -145.4268799, 122.4062500, -260.0170593, 261.3128662
8: -165.0796051, 111.9448547, -174.3259735, 118.1979218, -283.2775269, 286.2708130
9: -124.9386139, 123.3972931, -131.9645691, 130.3583679, -255.2969818, 255.3618622

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 65

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1100243, upper bound: 326.1119683
time: 8.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1543842, upper bound: 326.1584190
time: 12.23 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1724833, upper bound: 326.1761376
time: 8.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 53.07 seconds
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 53.07
Output dim: 7, lower bound: -326.1778657, upper bound: 326.1805101
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 53.07
Output dim: 7, lower bound: -326.1778657, upper bound: 326.1871089
IS_A1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 53.07
Output dim: 7, lower bound: -326.1699950, upper bound: 326.1728023
IS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 53.07
Output dim: 7, lower bound: -326.1764140, upper bound: 326.1802217
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 53.07
Output dim: 7, lower bound: -326.1744239, upper bound: 326.1787699
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 53.07
Output dim: 7, lower bound: -326.1744239, upper bound: 326.1813699
IS_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 53.07
Output dim: 7, lower bound: -326.1543842, upper bound: 326.1584190
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 53.07
Output dim: 7, lower bound: -326.1724833, upper bound: 326.1761376
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 53.07
Output dim: 7, lower bound: -326.1751921, upper bound: 326.1787080
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 53.07
Output dim: 7, lower bound: -326.1830086, upper bound: 326.1892913
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 53.07
Output dim: 7, lower bound: -326.1729003, upper bound: 326.1732440
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 53.07
Output dim: 7, lower bound: -326.1728975, upper bound: 326.1742997
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=328.3682861328125
rel_dist={7: [-326.2560128858547, 326.2560128858547]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1948223, upper bound: 326.1959230
time: 14.92 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1914931, upper bound: 326.1914931
time: 7.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 22.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 22.92
Output dim: 7, lower bound: -326.1948223, upper bound: 326.1959230
IS_A2, status: Status.UNKNOWN, split count: 1, time: 22.92
Output dim: 7, lower bound: -326.1914931, upper bound: 326.1914931

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -176.1383057, 139.9498596, -176.8887177, 140.5452881, -316.6835632, 316.8385620
1: -148.1291351, 124.6177597, -148.7599487, 125.1486740, -273.2777710, 273.3777161
2: -194.3291779, 127.1356506, -195.1577606, 127.6752167, -322.0043945, 322.2933655
3: -206.5952301, 109.2217331, -207.4779510, 109.6864548, -316.2816772, 316.6996765
4: -188.8157654, 145.2543488, -189.6262207, 145.8749542, -334.6907349, 334.8805542
5: -169.4739380, 132.2519989, -170.1939697, 132.8175659, -302.2914734, 302.4459229
6: -162.5175018, 156.2819061, -163.2100983, 156.9458160, -319.4632874, 319.4920044
7: -177.6271973, 149.3514099, -178.3847504, 149.9835510, -327.6107178, 327.7361450
8: -212.9792480, 144.7171021, -213.8840027, 145.3365479, -358.3157654, 358.6010742
9: -161.1715698, 159.2395630, -161.8587646, 159.9163361, -321.0878906, 321.0983276

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1914931, upper bound: 326.1914931
time: 8.28 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1914931, upper bound: 326.1914931
time: 8.49 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -170.4180756, 135.3389435, -170.1840973, 135.2151794, -305.6332092, 305.5230408
1: -143.2613831, 120.4438782, -143.1229401, 120.3964691, -263.6578369, 263.5668335
2: -187.9421997, 122.8000488, -187.7509918, 122.8326645, -310.7748108, 310.5509949
3: -199.7481079, 105.6107330, -199.5851288, 105.5312500, -305.2793274, 305.1958008
4: -182.5223236, 140.3965607, -182.3829193, 140.3240967, -322.8464355, 322.7794495
5: -163.9718323, 127.7645874, -163.7547760, 127.7495499, -291.7213745, 291.5193481
6: -157.1690369, 151.1747589, -157.0173645, 151.0122681, -308.1813049, 308.1920776
7: -171.7068329, 144.4490356, -171.6126099, 144.3276825, -316.0344849, 316.0616455
8: -206.0433807, 139.8561096, -205.7956696, 139.7927856, -345.8361511, 345.6517334
9: -155.8437653, 153.9222107, -155.7152252, 153.8632965, -309.7070312, 309.6373901

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1914931, upper bound: 326.1914931
time: 8.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1914931, upper bound: 326.1914931
time: 8.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.26 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 7, lower bound: -326.1914931, upper bound: 326.1914931
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 7, lower bound: -326.1914931, upper bound: 326.1914931
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 7, lower bound: -326.1914931, upper bound: 326.1914931
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 7, lower bound: -326.1914931, upper bound: 326.1914931

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -176.1383057, 139.9498596, -176.1383057, 139.9498596, -316.0881653, 316.0881653
1: -148.1291351, 124.6177597, -148.1291351, 124.6177597, -272.7468872, 272.7468872
2: -194.3291779, 127.1356506, -194.3291779, 127.1356506, -321.4647827, 321.4647827
3: -206.5952301, 109.2217331, -206.5952301, 109.2217331, -315.8169250, 315.8169250
4: -188.8157654, 145.2543488, -188.8157654, 145.2543488, -334.0700989, 334.0700989
5: -169.4739380, 132.2519989, -169.4739380, 132.2519989, -301.7259216, 301.7259216
6: -162.5175018, 156.2819061, -162.5175018, 156.2819061, -318.7994080, 318.7994080
7: -177.6271973, 149.3514099, -177.6271973, 149.3514099, -326.9786072, 326.9786072
8: -212.9792480, 144.7171021, -212.9792480, 144.7171021, -357.6962891, 357.6962891
9: -161.1715698, 159.2395630, -161.1715698, 159.2395630, -320.4111328, 320.4111328

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1941297, upper bound: 326.1952322
time: 11.23 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1941545, upper bound: 326.1952689
time: 11.11 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -176.1383057, 139.9498596, -170.4180756, 135.3389435, -311.4772339, 310.3679199
1: -148.1291351, 124.6177597, -143.2613831, 120.4438782, -268.5729980, 267.8791504
2: -194.3291779, 127.1356506, -187.9421997, 122.8000488, -317.1292114, 315.0777893
3: -206.5952301, 109.2217331, -199.7481079, 105.6107330, -312.2059326, 308.9698181
4: -188.8157654, 145.2543488, -182.5223236, 140.3965607, -329.2122803, 327.7766724
5: -169.4739380, 132.2519989, -163.9718323, 127.7645874, -297.2384949, 296.2238159
6: -162.5175018, 156.2819061, -157.1690369, 151.1747589, -313.6921692, 313.4509277
7: -177.6271973, 149.3514099, -171.7068329, 144.4490356, -322.0762329, 321.0582275
8: -212.9792480, 144.7171021, -206.0433807, 139.8561096, -352.8352966, 350.7604980
9: -161.1715698, 159.2395630, -155.8437653, 153.9222107, -315.0937195, 315.0833130

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764128, upper bound: 326.1776470
time: 12.83 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1784531, upper bound: 326.1799809
time: 11.97 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -170.4180756, 135.3389435, -176.1383057, 139.9498596, -310.3679199, 311.4772339
1: -143.2613831, 120.4438782, -148.1291351, 124.6177597, -267.8791504, 268.5729980
2: -187.9421997, 122.8000488, -194.3291779, 127.1356506, -315.0777893, 317.1292114
3: -199.7481079, 105.6107330, -206.5952301, 109.2217331, -308.9698181, 312.2059326
4: -182.5223236, 140.3965607, -188.8157654, 145.2543488, -327.7766724, 329.2122803
5: -163.9718323, 127.7645874, -169.4739380, 132.2519989, -296.2238159, 297.2384949
6: -157.1690369, 151.1747589, -162.5175018, 156.2819061, -313.4509277, 313.6921692
7: -171.7068329, 144.4490356, -177.6271973, 149.3514099, -321.0582275, 322.0762329
8: -206.0433807, 139.8561096, -212.9792480, 144.7171021, -350.7604980, 352.8352966
9: -155.8437653, 153.9222107, -161.1715698, 159.2395630, -315.0833130, 315.0937195

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1727972, upper bound: 326.1728857
time: 8.35 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1742277, upper bound: 326.1742277
time: 7.71 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -170.4180756, 135.3389435, -170.4180756, 135.3389435, -305.7570190, 305.7570190
1: -143.2613831, 120.4438782, -143.2613831, 120.4438782, -263.7052612, 263.7052612
2: -187.9421997, 122.8000488, -187.9421997, 122.8000488, -310.7422180, 310.7422180
3: -199.7481079, 105.6107330, -199.7481079, 105.6107330, -305.3588257, 305.3588257
4: -182.5223236, 140.3965607, -182.5223236, 140.3965607, -322.9188843, 322.9188843
5: -163.9718323, 127.7645874, -163.9718323, 127.7645874, -291.7364197, 291.7364197
6: -157.1690369, 151.1747589, -157.1690369, 151.1747589, -308.3437500, 308.3437500
7: -171.7068329, 144.4490356, -171.7068329, 144.4490356, -316.1558838, 316.1558838
8: -206.0433807, 139.8561096, -206.0433807, 139.8561096, -345.8994751, 345.8994751
9: -155.8437653, 153.9222107, -155.8437653, 153.9222107, -309.7659607, 309.7659607

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1909034, upper bound: 326.1909261
time: 10.69 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
time: 9.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.95 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.95
Output dim: 7, lower bound: -326.1941297, upper bound: 326.1952322
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.95
Output dim: 7, lower bound: -326.1941545, upper bound: 326.1952689
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.95
Output dim: 7, lower bound: -326.1764128, upper bound: 326.1776470
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.95
Output dim: 7, lower bound: -326.1784531, upper bound: 326.1799809
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 25.95
Output dim: 7, lower bound: -326.1727972, upper bound: 326.1728857
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 25.95
Output dim: 7, lower bound: -326.1742277, upper bound: 326.1742277
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.95
Output dim: 7, lower bound: -326.1909034, upper bound: 326.1909261
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.95
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -172.4068756, 137.0090790, -173.2606812, 137.6778107, -310.0846863, 310.2697754
1: -144.9750977, 121.9808197, -145.7066345, 122.5920944, -267.5671997, 267.6874390
2: -190.2096252, 124.4830017, -191.1572418, 125.1017990, -315.3113708, 315.6402588
3: -202.2120056, 106.9094696, -203.2171326, 107.4488678, -309.6607666, 310.1265869
4: -184.7848053, 142.2056732, -185.7175903, 142.9009094, -327.6857300, 327.9232178
5: -165.9068451, 129.4634552, -166.7141876, 130.1047516, -296.0115967, 296.1776428
6: -159.0729065, 152.9787445, -159.8703308, 153.7397003, -312.8125916, 312.8490601
7: -173.8363190, 146.1954803, -174.7204132, 146.9294434, -320.7657471, 320.9158630
8: -208.4698792, 141.6994171, -209.5016785, 142.3826141, -350.8524780, 351.2011108
9: -157.7527313, 155.8794403, -158.5487671, 156.6630096, -314.4157104, 314.4282227

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2558140, upper bound: 326.2558140
time: 9.99 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2558140, upper bound: 326.2558157
time: 10.11 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -173.7284241, 138.0476227, -174.3557739, 138.5428772, -312.2712708, 312.4033813
1: -146.1010895, 122.9223862, -146.6289673, 123.3637085, -269.4647522, 269.5513611
2: -191.6738434, 125.4348755, -192.3651581, 125.8777084, -317.5514832, 317.8000488
3: -203.7657166, 107.7374496, -204.5022888, 108.1238785, -311.8895874, 312.2396851
4: -186.2195435, 143.2833252, -186.8953400, 143.7964020, -330.0159302, 330.1786499
5: -167.1637726, 130.4519043, -167.7652893, 130.9206543, -298.0844116, 298.2171631
6: -160.3018341, 154.1531677, -160.8785095, 154.7072601, -315.0090942, 315.0316772
7: -175.1910095, 147.3228149, -175.8252563, 147.8510132, -323.0420227, 323.1480713
8: -210.0689392, 142.7631989, -210.8265076, 143.2718506, -353.3407898, 353.5897217
9: -158.9735107, 157.0806885, -159.5457001, 157.6426544, -316.6161499, 316.6263733

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2558157, upper bound: 326.2558206
time: 11.37 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2558157, upper bound: 326.2558423
time: 11.45 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -142.2155457, 113.0665283, -146.4197693, 116.3174820, -258.5329895, 259.4862366
1: -119.5971680, 100.6445160, -123.0648575, 103.4762650, -223.0734100, 223.7093658
2: -156.8995361, 102.8950119, -161.4568787, 105.6455383, -262.5450745, 264.3518677
3: -166.7689209, 88.2479248, -171.5615997, 90.7664108, -257.5353394, 259.8095093
4: -152.2905579, 117.3294754, -156.6774597, 120.6315842, -272.9220886, 274.0069275
5: -136.7775879, 106.7972031, -140.8257751, 109.7509613, -246.5285187, 247.6229706
6: -131.3273315, 126.2379913, -135.1081696, 129.9142151, -261.2415466, 261.3461609
7: -143.5122375, 120.8269882, -147.5700684, 124.2620926, -267.7743225, 268.3970642
8: -172.1525116, 116.8194199, -177.1630707, 120.1146622, -292.2671814, 293.9824829
9: -130.2881470, 128.6931000, -133.9959106, 132.3088989, -262.5970459, 262.6889954

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1763920, upper bound: 326.1776470
time: 12.30 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1763920, upper bound: 326.1776470
time: 11.10 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -161.7462463, 128.5256042, -159.4005890, 126.5946274, -288.3408508, 287.9261475
1: -136.0521088, 114.4738770, -134.0136414, 112.6753082, -248.7274170, 248.4875031
2: -178.4775238, 116.8844299, -175.8009186, 114.9418793, -293.4194031, 292.6853638
3: -189.7207489, 100.3388824, -186.8314209, 98.8093872, -288.5301514, 287.1702576
4: -173.3217468, 133.4183502, -170.6549683, 131.3316345, -304.6533813, 304.0733032
5: -155.6003876, 121.4683685, -153.3539734, 119.5021896, -275.1025696, 274.8223267
6: -149.2780762, 143.5605011, -147.0319366, 141.4327393, -290.7108154, 290.5924072
7: -163.2038574, 137.2861786, -160.6509552, 135.2063599, -298.4102173, 297.9371338
8: -195.6663208, 132.8450928, -192.7862244, 130.7706909, -326.4370117, 325.6312866
9: -148.0830536, 146.3120728, -145.8195801, 144.0211182, -292.1041260, 292.1316528

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1771316, upper bound: 326.1787114
time: 12.26 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1771316, upper bound: 326.1799809
time: 10.59 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -159.4005890, 126.5946274, -161.7462463, 128.5256042, -287.9261475, 288.3408508
1: -134.0136414, 112.6753082, -136.0521088, 114.4738770, -248.4875031, 248.7274170
2: -175.8009186, 114.9418793, -178.4775238, 116.8844299, -292.6853638, 293.4194031
3: -186.8314209, 98.8093872, -189.7207489, 100.3388824, -287.1702576, 288.5301514
4: -170.6549683, 131.3316345, -173.3217468, 133.4183502, -304.0733032, 304.6533813
5: -153.3539734, 119.5021896, -155.6003876, 121.4683685, -274.8223267, 275.1025696
6: -147.0319366, 141.4327393, -149.2780762, 143.5605011, -290.5924072, 290.7108154
7: -160.6509552, 135.2063599, -163.2038574, 137.2861786, -297.9371338, 298.4102173
8: -192.7862244, 130.7706909, -195.6663208, 132.8450928, -325.6312866, 326.4370117
9: -145.8195801, 144.0211182, -148.0830536, 146.3120728, -292.1316528, 292.1041260

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1787114, upper bound: 326.1771316
time: 10.06 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1787114, upper bound: 326.1784531
time: 12.41 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -166.5475159, 132.2863159, -167.4389038, 132.9868927, -299.5343933, 299.7252197
1: -139.9825134, 117.7055435, -140.7507629, 118.3453217, -258.3278198, 258.4562988
2: -183.6661987, 120.0485992, -184.6576843, 120.6933517, -304.3595581, 304.7062988
3: -195.2027588, 103.2172470, -196.2506409, 103.7759705, -298.9787292, 299.4678955
4: -178.3370361, 137.2300415, -179.3115387, 137.9587555, -316.2957764, 316.5415649
5: -160.2666626, 124.8754883, -161.1144562, 125.5436478, -285.8103027, 285.9899292
6: -153.5978241, 147.7451935, -154.4299469, 148.5403900, -302.1381836, 302.1751404
7: -167.7770691, 141.1808014, -168.6988525, 141.9430084, -309.7200928, 309.8796387
8: -201.3616943, 136.7145081, -202.4421844, 137.4345245, -338.7962036, 339.1566772
9: -152.3013611, 150.4405060, -153.1302643, 151.2554016, -303.5567627, 303.5707397

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
time: 8.64 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
time: 9.59 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -167.8958130, 133.3486938, -168.5579071, 133.8711853, -301.7669678, 301.9066162
1: -141.1369476, 118.6681290, -141.6945190, 119.1342621, -260.2712097, 260.3626404
2: -185.1630707, 121.0185623, -185.8926392, 121.4862366, -306.6492920, 306.9111633
3: -196.7878113, 104.0571823, -197.5649261, 104.4650269, -301.2528381, 301.6221008
4: -179.8021088, 138.3326874, -180.5161896, 138.8744507, -318.6765747, 318.8488770
5: -161.5542755, 125.8829346, -162.1889801, 126.3769150, -287.9311829, 288.0718994
6: -154.8511200, 148.9445190, -155.4595642, 149.5299683, -304.3810730, 304.4040833
7: -169.1579437, 142.3271179, -169.8270569, 142.8841095, -312.0420532, 312.1541748
8: -202.9967346, 137.8070679, -203.7964783, 138.3449860, -341.3417358, 341.6035156
9: -153.5454102, 151.6636963, -154.1486816, 152.2565002, -305.8017883, 305.8123474

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
time: 8.81 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
time: 9.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.33 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.2558140, upper bound: 326.2558140
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.2558140, upper bound: 326.2558157
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.2558157, upper bound: 326.2558206
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.2558157, upper bound: 326.2558423
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.1763920, upper bound: 326.1776470
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.1763920, upper bound: 326.1776470
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.1771316, upper bound: 326.1787114
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.1771316, upper bound: 326.1799809
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.1787114, upper bound: 326.1771316
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.1787114, upper bound: 326.1784531
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -172.4068756, 137.0090790, -172.4068756, 137.0090790, -309.4159546, 309.4159546
1: -144.9750977, 121.9808197, -144.9750977, 121.9808197, -266.9559326, 266.9559326
2: -190.2096252, 124.4830017, -190.2096252, 124.4830017, -314.6925964, 314.6925964
3: -202.2120056, 106.9094696, -202.2120056, 106.9094696, -309.1214600, 309.1214600
4: -184.7848053, 142.2056732, -184.7848053, 142.2056732, -326.9904785, 326.9904785
5: -165.9068451, 129.4634552, -165.9068451, 129.4634552, -295.3702698, 295.3702698
6: -159.0729065, 152.9787445, -159.0729065, 152.9787445, -312.0516357, 312.0516357
7: -173.8363190, 146.1954803, -173.8363190, 146.1954803, -320.0317688, 320.0317688
8: -208.4698792, 141.6994171, -208.4698792, 141.6994171, -350.1693115, 350.1693115
9: -157.7527313, 155.8794403, -157.7527313, 155.8794403, -313.6321716, 313.6321716

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2361769, upper bound: 326.2364328
time: 11.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2352526, upper bound: 326.2352526
time: 9.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -172.4068756, 137.0090790, -173.7010040, 138.0261688, -310.4330444, 310.7100830
1: -144.9750977, 121.9808197, -146.0771637, 122.9020615, -267.8771667, 268.0579834
2: -190.2096252, 124.4830017, -191.6437531, 125.4152222, -315.6247559, 316.1267395
3: -202.2120056, 106.9094696, -203.7337036, 107.7204285, -309.9323730, 310.6431580
4: -184.7848053, 142.2056732, -186.1897888, 143.2603455, -328.0451355, 328.3954468
5: -165.9068451, 129.4634552, -167.1373901, 130.4316711, -296.3384399, 296.6008301
6: -159.0729065, 152.9787445, -160.2767029, 154.1280823, -313.2009583, 313.2554321
7: -173.8363190, 146.1954803, -175.1632996, 147.2997131, -321.1360474, 321.3587341
8: -208.4698792, 141.6994171, -210.0354767, 142.7402802, -351.2101440, 351.7348938
9: -157.7527313, 155.8794403, -158.9484863, 157.0557861, -314.8085022, 314.8279419

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2364328, upper bound: 326.2361785
time: 12.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2352526, upper bound: 326.2352643
time: 8.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -173.7284241, 138.0476227, -172.4068756, 137.0090790, -310.7374573, 310.4544983
1: -146.1010895, 122.9223862, -144.9750977, 121.9808197, -268.0818787, 267.8974915
2: -191.6738434, 125.4348755, -190.2096252, 124.4830017, -316.1568298, 315.6444092
3: -203.7657166, 107.7374496, -202.2120056, 106.9094696, -310.6751709, 309.9493713
4: -186.2195435, 143.2833252, -184.7848053, 142.2056732, -328.4252319, 328.0681152
5: -167.1637726, 130.4519043, -165.9068451, 129.4634552, -296.6272278, 296.3587036
6: -160.3018341, 154.1531677, -159.0729065, 152.9787445, -313.2805786, 313.2260132
7: -175.1910095, 147.3228149, -173.8363190, 146.1954803, -321.3864441, 321.1591187
8: -210.0689392, 142.7631989, -208.4698792, 141.6994171, -351.7683716, 351.2330933
9: -158.9735107, 157.0806885, -157.7527313, 155.8794403, -314.8529358, 314.8334045

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2361785, upper bound: 326.2364396
time: 11.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2352645, upper bound: 326.2352849
time: 8.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -173.7284241, 138.0476227, -173.7284241, 138.0476227, -311.7760620, 311.7760620
1: -146.1010895, 122.9223862, -146.1010895, 122.9223862, -269.0234680, 269.0234680
2: -191.6738434, 125.4348755, -191.6738434, 125.4348755, -317.1086426, 317.1086426
3: -203.7657166, 107.7374496, -203.7657166, 107.7374496, -311.5031433, 311.5031433
4: -186.2195435, 143.2833252, -186.2195435, 143.2833252, -329.5028687, 329.5028687
5: -167.1637726, 130.4519043, -167.1637726, 130.4519043, -297.6156616, 297.6156616
6: -160.3018341, 154.1531677, -160.3018341, 154.1531677, -314.4549866, 314.4549866
7: -175.1910095, 147.3228149, -175.1910095, 147.3228149, -322.5138245, 322.5138245
8: -210.0689392, 142.7631989, -210.0689392, 142.7631989, -352.8321228, 352.8321228
9: -158.9735107, 157.0806885, -158.9735107, 157.0806885, -316.0541687, 316.0541687

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2364864, upper bound: 326.2364585
time: 11.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2352643, upper bound: 326.2355295
time: 7.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -142.2155457, 113.0665283, -138.0074310, 109.6539154, -251.8694458, 251.0739441
1: -119.5971680, 100.6445160, -115.9966431, 97.5335312, -217.1306915, 216.6411591
2: -156.8995361, 102.8950119, -152.1828766, 99.6459579, -256.5455017, 255.0778809
3: -166.7689209, 88.2479248, -161.6824646, 85.5678101, -252.3367310, 249.9303894
4: -152.2905579, 117.3294754, -147.6199341, 113.7110596, -266.0014954, 264.9494019
5: -136.7775879, 106.7972031, -132.7144318, 103.4450531, -240.2226410, 239.5116272
6: -131.3273315, 126.2379913, -127.3721695, 122.4666061, -253.7939453, 253.6101685
7: -143.5122375, 120.8269882, -139.1227875, 117.1959381, -260.7081604, 259.9497681
8: -172.1525116, 116.8194199, -167.0466156, 113.1972351, -285.3497314, 283.8660278
9: -130.2881470, 128.6931000, -126.3391037, 124.7404633, -255.0286102, 255.0321960

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1662502, upper bound: 326.1672686
time: 10.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1679436, upper bound: 326.1691396
time: 12.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -142.2155457, 113.0665283, -155.4663696, 123.4709396, -265.6864624, 268.5328369
1: -119.5971680, 100.6445160, -130.7114105, 109.9020538, -229.4991913, 231.3559265
2: -156.8995361, 102.8950119, -171.4671173, 112.1376038, -269.0371094, 274.3621216
3: -166.7689209, 88.2479248, -182.2185059, 96.3812790, -263.1501770, 270.4664307
4: -152.2905579, 117.3294754, -166.4190674, 128.0948944, -280.3854065, 283.7485352
5: -136.7775879, 106.7972031, -149.5622101, 116.5533905, -253.3309784, 256.3594055
6: -131.3273315, 126.2379913, -143.4135742, 137.9552765, -269.2825928, 269.6515503
7: -143.5122375, 120.8269882, -156.7063904, 131.9080963, -275.4203491, 277.5333862
8: -172.1525116, 116.8194199, -188.0544586, 127.5249329, -299.6774292, 304.8738708
9: -130.2881470, 128.6931000, -142.2408295, 140.4872742, -270.7754211, 270.9338684

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1652475, upper bound: 326.1664610
time: 10.30 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1679436, upper bound: 326.1691396
time: 12.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -161.7462463, 128.5256042, -138.0074310, 109.6539154, -271.4001465, 266.5330200
1: -136.0521088, 114.4738770, -115.9966431, 97.5335312, -233.5856323, 230.4704895
2: -178.4775238, 116.8844299, -152.1828766, 99.6459579, -278.1234741, 269.0673218
3: -189.7207489, 100.3388824, -161.6824646, 85.5678101, -275.2885437, 262.0213013
4: -173.3217468, 133.4183502, -147.6199341, 113.7110596, -287.0327454, 281.0382690
5: -155.6003876, 121.4683685, -132.7144318, 103.4450531, -259.0454407, 254.1828003
6: -149.2780762, 143.5605011, -127.3721695, 122.4666061, -271.7446899, 270.9325867
7: -163.2038574, 137.2861786, -139.1227875, 117.1959381, -280.3997498, 276.4089661
8: -195.6663208, 132.8450928, -167.0466156, 113.1972351, -308.8635559, 299.8917236
9: -148.0830536, 146.3120728, -126.3391037, 124.7404633, -272.8234863, 272.6511841

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1668927, upper bound: 326.1681910
time: 9.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1687054, upper bound: 326.1702020
time: 12.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -161.7462463, 128.5256042, -155.4737396, 123.4770508, -285.2232361, 283.9992676
1: -136.0521088, 114.4738770, -130.7179871, 109.9071198, -245.9592133, 245.1918335
2: -178.4775238, 116.8844299, -171.4756317, 112.1434097, -290.6209412, 288.3600464
3: -189.7207489, 100.3388824, -182.2270660, 96.3855667, -286.1063232, 282.5659485
4: -173.3217468, 133.4183502, -166.4269257, 128.1009521, -301.4226990, 299.8452759
5: -155.6003876, 121.4683685, -149.5692749, 116.5590286, -272.1593933, 271.0376587
6: -149.2780762, 143.5605011, -143.4204407, 137.9618683, -287.2399292, 286.9808655
7: -163.2038574, 137.2861786, -156.7140503, 131.9145813, -295.1183777, 294.0002441
8: -195.6663208, 132.8450928, -188.0634613, 127.5310440, -323.1973572, 320.9085693
9: -148.0830536, 146.3120728, -142.2478027, 140.4941254, -288.5771790, 288.5598450

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 65

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1640751, upper bound: 326.1649932
time: 11.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1771316, upper bound: 326.1799809
time: 11.98 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -138.0074310, 109.6539154, -161.7462463, 128.5256042, -266.5330200, 271.4001465
1: -115.9966431, 97.5335312, -136.0521088, 114.4738770, -230.4704895, 233.5856323
2: -152.1828766, 99.6459579, -178.4775238, 116.8844299, -269.0673218, 278.1234741
3: -161.6824646, 85.5678101, -189.7207489, 100.3388824, -262.0213013, 275.2885437
4: -147.6199341, 113.7110596, -173.3217468, 133.4183502, -281.0382690, 287.0327454
5: -132.7144318, 103.4450531, -155.6003876, 121.4683685, -254.1828003, 259.0454407
6: -127.3721695, 122.4666061, -149.2780762, 143.5605011, -270.9325867, 271.7446899
7: -139.1227875, 117.1959381, -163.2038574, 137.2861786, -276.4089661, 280.3997498
8: -167.0466156, 113.1972351, -195.6663208, 132.8450928, -299.8917236, 308.8635559
9: -126.3391037, 124.7404633, -148.0830536, 146.3120728, -272.6511841, 272.8234863

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1672686, upper bound: 326.1668927
time: 11.54 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1691396, upper bound: 326.1687054
time: 12.32 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -155.4737396, 123.4770508, -161.7462463, 128.5256042, -283.9992676, 285.2232361
1: -130.7179871, 109.9071198, -136.0521088, 114.4738770, -245.1918335, 245.9592133
2: -171.4756317, 112.1434097, -178.4775238, 116.8844299, -288.3600464, 290.6209412
3: -182.2270660, 96.3855667, -189.7207489, 100.3388824, -282.5659485, 286.1063232
4: -166.4269257, 128.1009521, -173.3217468, 133.4183502, -299.8452759, 301.4226990
5: -149.5692749, 116.5590286, -155.6003876, 121.4683685, -271.0376587, 272.1593933
6: -143.4204407, 137.9618683, -149.2780762, 143.5605011, -286.9808655, 287.2399292
7: -156.7140503, 131.9145813, -163.2038574, 137.2861786, -294.0002441, 295.1183777
8: -188.0634613, 127.5310440, -195.6663208, 132.8450928, -320.9085693, 323.1973572
9: -142.2478027, 140.4941254, -148.0830536, 146.3120728, -288.5598450, 288.5771790

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1644147, upper bound: 326.1640340
time: 12.22 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1776470, upper bound: 326.1779718
time: 10.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -166.5475159, 132.2863159, -166.5475159, 132.2863159, -298.8338318, 298.8338318
1: -139.9825134, 117.7055435, -139.9825134, 117.7055435, -257.6880493, 257.6880493
2: -183.6661987, 120.0485992, -183.6661987, 120.0485992, -303.7147827, 303.7147827
3: -195.2027588, 103.2172470, -195.2027588, 103.2172470, -298.4200134, 298.4200134
4: -178.3370361, 137.2300415, -178.3370361, 137.2300415, -315.5670776, 315.5670776
5: -160.2666626, 124.8754883, -160.2666626, 124.8754883, -285.1421509, 285.1421509
6: -153.5978241, 147.7451935, -153.5978241, 147.7451935, -301.3430176, 301.3430176
7: -167.7770691, 141.1808014, -167.7770691, 141.1808014, -308.9578552, 308.9578552
8: -201.3616943, 136.7145081, -201.3616943, 136.7145081, -338.0762024, 338.0762024
9: -152.3013611, 150.4405060, -152.3013611, 150.4405060, -302.7418823, 302.7418823

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 189
type: B, layer: 1, pos: 189
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 224
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1740357, upper bound: 326.1735450
time: 9.23 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1909034, upper bound: 326.1909261
time: 9.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 32.68 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.2361769, upper bound: 326.2364328
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.2352526, upper bound: 326.2352526
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.2364328, upper bound: 326.2361785
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.2352526, upper bound: 326.2352643
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.2361785, upper bound: 326.2364396
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.2352645, upper bound: 326.2352849
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.2364864, upper bound: 326.2364585
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.2352643, upper bound: 326.2355295
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1662502, upper bound: 326.1672686
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1679436, upper bound: 326.1691396
IS_A1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1652475, upper bound: 326.1664610
IS_A1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1679436, upper bound: 326.1691396
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1668927, upper bound: 326.1681910
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1687054, upper bound: 326.1702020
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1640751, upper bound: 326.1649932
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1771316, upper bound: 326.1799809
IS_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1672686, upper bound: 326.1668927
IS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1691396, upper bound: 326.1687054
IS_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1644147, upper bound: 326.1640340
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1776470, upper bound: 326.1779718
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1740357, upper bound: 326.1735450
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.68
Output dim: 7, lower bound: -326.1909034, upper bound: 326.1909261
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.68
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.68
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.68
Output dim: 7, lower bound: -326.1908623, upper bound: 326.1908623
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=328.3682861328125
rel_dist={7: [-326.25584232239004, 326.2558422835341]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1831.01 seconds
