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
execution time: IAR + LP analysis = 1.21 + 10.94 = 12.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -326.2561776, upper bound: 326.2561776


# Binary Search by BASE starts (time budget: 2687.85 seconds, max iter: 100)

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
Binary search time: 44.63 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2643.22 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1916368, upper bound: 326.1916367
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1916368, upper bound: 326.1916367
time: 6.14 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.42
Output dim: 7, lower bound: -326.1916368, upper bound: 326.1916367
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.42
Output dim: 7, lower bound: -326.1916368, upper bound: 326.1916367

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743237, upper bound: 326.1743231
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743231, upper bound: 326.1743237
time: 5.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743237, upper bound: 326.1743231
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743231, upper bound: 326.1743237
time: 5.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.17 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.17
Output dim: 7, lower bound: -326.1743237, upper bound: 326.1743231
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.17
Output dim: 7, lower bound: -326.1743231, upper bound: 326.1743237
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.17
Output dim: 7, lower bound: -326.1743237, upper bound: 326.1743231
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.17
Output dim: 7, lower bound: -326.1743231, upper bound: 326.1743237

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731388, upper bound: 326.1731363
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731363, upper bound: 326.1731373
time: 6.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731373, upper bound: 326.1731363
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731363, upper bound: 326.1731388
time: 8.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731388, upper bound: 326.1731363
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731363, upper bound: 326.1731373
time: 6.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731373, upper bound: 326.1731363
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731363, upper bound: 326.1731388
time: 8.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.83 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 7, lower bound: -326.1731388, upper bound: 326.1731363
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 7, lower bound: -326.1731363, upper bound: 326.1731373
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 7, lower bound: -326.1731373, upper bound: 326.1731363
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 7, lower bound: -326.1731363, upper bound: 326.1731388
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 7, lower bound: -326.1731388, upper bound: 326.1731363
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 7, lower bound: -326.1731363, upper bound: 326.1731373
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 7, lower bound: -326.1731373, upper bound: 326.1731363
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.83
Output dim: 7, lower bound: -326.1731363, upper bound: 326.1731388

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729072, upper bound: 326.1729082
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729072, upper bound: 326.1729082
time: 5.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729092, upper bound: 326.1729058
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729092, upper bound: 326.1729058
time: 6.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729058, upper bound: 326.1729092
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729058, upper bound: 326.1729092
time: 6.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729082, upper bound: 326.1729072
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729082, upper bound: 326.1729072
time: 5.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729072, upper bound: 326.1729082
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729072, upper bound: 326.1729082
time: 5.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729092, upper bound: 326.1729058
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729092, upper bound: 326.1729058
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729058, upper bound: 326.1729092
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729058, upper bound: 326.1729092
time: 6.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729082, upper bound: 326.1729072
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729082, upper bound: 326.1729072
time: 5.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729072, upper bound: 326.1729082
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729072, upper bound: 326.1729082
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729092, upper bound: 326.1729058
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729092, upper bound: 326.1729058
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729058, upper bound: 326.1729092
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729058, upper bound: 326.1729092
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729082, upper bound: 326.1729072
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729082, upper bound: 326.1729072
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729072, upper bound: 326.1729082
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729072, upper bound: 326.1729082
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729092, upper bound: 326.1729058
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729092, upper bound: 326.1729058
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729058, upper bound: 326.1729092
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729058, upper bound: 326.1729092
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729082, upper bound: 326.1729072
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.99
Output dim: 7, lower bound: -326.1729082, upper bound: 326.1729072
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=328.3682861328125
rel_dist={7: [-326.25613672106726, 326.2561367077651]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1916479, upper bound: 326.1916479
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1916479, upper bound: 326.1916479
time: 5.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.22
Output dim: 7, lower bound: -326.1916479, upper bound: 326.1916479
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.22
Output dim: 7, lower bound: -326.1916479, upper bound: 326.1916479

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743469, upper bound: 326.1743463
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743463, upper bound: 326.1743469
time: 7.19 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743469, upper bound: 326.1743463
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743463, upper bound: 326.1743469
time: 7.15 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.05
Output dim: 7, lower bound: -326.1743469, upper bound: 326.1743463
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.05
Output dim: 7, lower bound: -326.1743463, upper bound: 326.1743469
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.05
Output dim: 7, lower bound: -326.1743469, upper bound: 326.1743463
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.05
Output dim: 7, lower bound: -326.1743463, upper bound: 326.1743469

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731610, upper bound: 326.1731573
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731572, upper bound: 326.1731599
time: 6.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731599, upper bound: 326.1731572
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731573, upper bound: 326.1731610
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731610, upper bound: 326.1731573
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731572, upper bound: 326.1731599
time: 6.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731599, upper bound: 326.1731572
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731573, upper bound: 326.1731610
time: 6.25 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 7, lower bound: -326.1731610, upper bound: 326.1731573
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 7, lower bound: -326.1731572, upper bound: 326.1731599
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 7, lower bound: -326.1731599, upper bound: 326.1731572
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 7, lower bound: -326.1731573, upper bound: 326.1731610
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 7, lower bound: -326.1731610, upper bound: 326.1731573
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 7, lower bound: -326.1731572, upper bound: 326.1731599
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 7, lower bound: -326.1731599, upper bound: 326.1731572
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 7, lower bound: -326.1731573, upper bound: 326.1731610

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729394, upper bound: 326.1729406
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729394, upper bound: 326.1729406
time: 7.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729413, upper bound: 326.1729377
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729413, upper bound: 326.1729377
time: 6.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729377, upper bound: 326.1729413
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729377, upper bound: 326.1729413
time: 7.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729406, upper bound: 326.1729394
time: 10.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729406, upper bound: 326.1729394
time: 8.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729394, upper bound: 326.1729406
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729393, upper bound: 326.1729406
time: 7.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729413, upper bound: 326.1729377
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729413, upper bound: 326.1729377
time: 6.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729377, upper bound: 326.1729413
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729377, upper bound: 326.1729413
time: 7.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729406, upper bound: 326.1729394
time: 10.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1729406, upper bound: 326.1729394
time: 8.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729394, upper bound: 326.1729406
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729394, upper bound: 326.1729406
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729413, upper bound: 326.1729377
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729413, upper bound: 326.1729377
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729377, upper bound: 326.1729413
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729377, upper bound: 326.1729413
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729406, upper bound: 326.1729394
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729406, upper bound: 326.1729394
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729394, upper bound: 326.1729406
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729393, upper bound: 326.1729406
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729413, upper bound: 326.1729377
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729413, upper bound: 326.1729377
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729377, upper bound: 326.1729413
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729377, upper bound: 326.1729413
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729406, upper bound: 326.1729394
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.03
Output dim: 7, lower bound: -326.1729406, upper bound: 326.1729394
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=328.3682861328125
rel_dist={7: [-326.25615739361524, 326.25615739361524]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1916553, upper bound: 326.1916553
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1916553, upper bound: 326.1916553
time: 5.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.29
Output dim: 7, lower bound: -326.1916553, upper bound: 326.1916553
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.29
Output dim: 7, lower bound: -326.1916553, upper bound: 326.1916553

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743620, upper bound: 326.1743611
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743611, upper bound: 326.1743620
time: 5.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743620, upper bound: 326.1743611
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743611, upper bound: 326.1743620
time: 5.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.03
Output dim: 7, lower bound: -326.1743620, upper bound: 326.1743611
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.03
Output dim: 7, lower bound: -326.1743611, upper bound: 326.1743620
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.03
Output dim: 7, lower bound: -326.1743620, upper bound: 326.1743611
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.03
Output dim: 7, lower bound: -326.1743611, upper bound: 326.1743620

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731752, upper bound: 326.1731704
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731696, upper bound: 326.1731747
time: 8.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731747, upper bound: 326.1731696
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731704, upper bound: 326.1731752
time: 6.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731752, upper bound: 326.1731704
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731696, upper bound: 326.1731747
time: 8.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731747, upper bound: 326.1731696
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731704, upper bound: 326.1731752
time: 6.41 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.07
Output dim: 7, lower bound: -326.1731752, upper bound: 326.1731704
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.07
Output dim: 7, lower bound: -326.1731696, upper bound: 326.1731747
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.07
Output dim: 7, lower bound: -326.1731747, upper bound: 326.1731696
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.07
Output dim: 7, lower bound: -326.1731704, upper bound: 326.1731752
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.07
Output dim: 7, lower bound: -326.1731752, upper bound: 326.1731704
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.07
Output dim: 7, lower bound: -326.1731696, upper bound: 326.1731747
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.07
Output dim: 7, lower bound: -326.1731747, upper bound: 326.1731696
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.07
Output dim: 7, lower bound: -326.1731704, upper bound: 326.1731752

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729591
time: 9.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729591
time: 8.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729581
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729581
time: 6.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729610
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729610
time: 7.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729600
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729600
time: 6.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729591
time: 8.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729591
time: 8.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729581
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729581
time: 6.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729610
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729610
time: 7.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729600
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729600
time: 6.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729591
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729591
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729581
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729581
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729610
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729610
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729600
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729600
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729591
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729591
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729581
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729581
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729610
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729610
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729600
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.43
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729600

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729158
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729212, upper bound: 326.1729591
time: 7.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729158
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729212, upper bound: 326.1729591
time: 5.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729581
time: 6.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729190, upper bound: 326.1729581
time: 6.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
time: 6.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
time: 7.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
time: 6.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
time: 6.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
time: 6.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729158
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729212, upper bound: 326.1729591
time: 6.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729158
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729212, upper bound: 326.1729591
time: 5.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729190, upper bound: 326.1729581
time: 6.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729190, upper bound: 326.1729581
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
time: 7.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
time: 6.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
time: 6.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
time: 6.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
time: 6.40 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 15.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729158
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729212, upper bound: 326.1729591
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729158
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729212, upper bound: 326.1729591
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729581
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729190, upper bound: 326.1729581
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729158
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729212, upper bound: 326.1729591
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729158
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729212, upper bound: 326.1729591
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729190, upper bound: 326.1729581
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729190, upper bound: 326.1729581
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.23
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677717, upper bound: 326.1677363
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677565, upper bound: 326.1677451
time: 5.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677598, upper bound: 326.1677583
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677367, upper bound: 326.1677655
time: 6.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677717, upper bound: 326.1677363
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677565, upper bound: 326.1677451
time: 6.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677598, upper bound: 326.1677583
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677367, upper bound: 326.1677655
time: 6.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677678, upper bound: 326.1677363
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677593, upper bound: 326.1677492
time: 6.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677462, upper bound: 326.1677557
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677363, upper bound: 326.1677695
time: 7.08 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 16.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677717, upper bound: 326.1677363
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677565, upper bound: 326.1677451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677598, upper bound: 326.1677583
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677367, upper bound: 326.1677655
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677717, upper bound: 326.1677363
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677565, upper bound: 326.1677451
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677598, upper bound: 326.1677583
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677367, upper bound: 326.1677655
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677678, upper bound: 326.1677363
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677593, upper bound: 326.1677492
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677462, upper bound: 326.1677557
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.72
Output dim: 7, lower bound: -326.1677363, upper bound: 326.1677695
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729190, upper bound: 326.1729581
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729158
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729212, upper bound: 326.1729591
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729600, upper bound: 326.1729158
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729212, upper bound: 326.1729591
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729190, upper bound: 326.1729581
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729610, upper bound: 326.1729158
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729190, upper bound: 326.1729581
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729581, upper bound: 326.1729190
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729591, upper bound: 326.1729212
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.72
Output dim: 7, lower bound: -326.1729158, upper bound: 326.1729600
Binary search (step 2): status=Status.UNKNOWN, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=328.3682861328125
rel_dist={7: [-326.2561708727637, 326.25617081751]}

## Binary search (step 3) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1916516, upper bound: 326.1916516
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1916516, upper bound: 326.1916516
time: 6.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.96 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.96
Output dim: 7, lower bound: -326.1916516, upper bound: 326.1916516
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.96
Output dim: 7, lower bound: -326.1916516, upper bound: 326.1916516

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743546, upper bound: 326.1743538
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743538, upper bound: 326.1743546
time: 6.27 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743546, upper bound: 326.1743538
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1743538, upper bound: 326.1743546
time: 6.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.65 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.65
Output dim: 7, lower bound: -326.1743546, upper bound: 326.1743538
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.65
Output dim: 7, lower bound: -326.1743538, upper bound: 326.1743546
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.65
Output dim: 7, lower bound: -326.1743546, upper bound: 326.1743538
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.65
Output dim: 7, lower bound: -326.1743538, upper bound: 326.1743546

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731681, upper bound: 326.1731639
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731634, upper bound: 326.1731673
time: 6.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731673, upper bound: 326.1731634
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731639, upper bound: 326.1731681
time: 6.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731681, upper bound: 326.1731639
time: 7.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731634, upper bound: 326.1731673
time: 7.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731673, upper bound: 326.1731634
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1731639, upper bound: 326.1731681
time: 6.50 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.26 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.26
Output dim: 7, lower bound: -326.1731681, upper bound: 326.1731639
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.26
Output dim: 7, lower bound: -326.1731634, upper bound: 326.1731673
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.26
Output dim: 7, lower bound: -326.1731673, upper bound: 326.1731634
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.26
Output dim: 7, lower bound: -326.1731639, upper bound: 326.1731681
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.26
Output dim: 7, lower bound: -326.1731681, upper bound: 326.1731639
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.26
Output dim: 7, lower bound: -326.1731634, upper bound: 326.1731673
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.26
Output dim: 7, lower bound: -326.1731673, upper bound: 326.1731634
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.26
Output dim: 7, lower bound: -326.1731639, upper bound: 326.1731681

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729509
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729509
time: 7.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729482
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729482
time: 6.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729516
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729516
time: 7.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729498
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729498
time: 6.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729509
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729509
time: 7.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729482
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729482
time: 6.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729516
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729516
time: 8.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729498
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729498
time: 6.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729509
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729509
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729482
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729482
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729516
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729516
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729498
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729498
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729509
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729509
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729482
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729482
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729516
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729516
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729498
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.84
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729498

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729103
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729149, upper bound: 326.1729509
time: 7.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729103
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729149, upper bound: 326.1729509
time: 6.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
time: 6.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
time: 7.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
time: 7.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
time: 6.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
time: 5.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
time: 7.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729103
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729149, upper bound: 326.1729509
time: 7.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729103
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729149, upper bound: 326.1729509
time: 6.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
time: 6.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
time: 7.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
time: 7.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
time: 6.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
time: 6.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
time: 7.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 17.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729103
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729149, upper bound: 326.1729509
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729103
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729149, upper bound: 326.1729509
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729103
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729149, upper bound: 326.1729509
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729103
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729149, upper bound: 326.1729509
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.05
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677668, upper bound: 326.1677339
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677526, upper bound: 326.1677410
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677539, upper bound: 326.1677548
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677344, upper bound: 326.1677621
time: 6.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677668, upper bound: 326.1677339
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677526, upper bound: 326.1677410
time: 5.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677539, upper bound: 326.1677548
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677344, upper bound: 326.1677621
time: 6.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677634, upper bound: 326.1677341
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677555, upper bound: 326.1677448
time: 6.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905
1: -148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693
2: -195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773
3: -207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982
4: -189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597
5: -170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746
6: -163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143
7: -178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861
8: -213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505
9: -161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677419, upper bound: 326.1677519
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1677339, upper bound: 326.1677652
time: 5.83 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 15.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677668, upper bound: 326.1677339
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677526, upper bound: 326.1677410
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677539, upper bound: 326.1677548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677344, upper bound: 326.1677621
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677668, upper bound: 326.1677339
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677526, upper bound: 326.1677410
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677539, upper bound: 326.1677548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677344, upper bound: 326.1677621
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677634, upper bound: 326.1677341
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677555, upper bound: 326.1677448
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677419, upper bound: 326.1677519
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.46
Output dim: 7, lower bound: -326.1677339, upper bound: 326.1677652
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729103
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729149, upper bound: 326.1729509
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729498, upper bound: 326.1729103
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729149, upper bound: 326.1729509
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729516, upper bound: 326.1729103
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729133, upper bound: 326.1729482
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729482, upper bound: 326.1729133
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729516
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729509, upper bound: 326.1729149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.46
Output dim: 7, lower bound: -326.1729103, upper bound: 326.1729498
Binary search (step 3): status=Status.UNKNOWN, k_low=10, k_high=10, k_mid=10, eps_mid=0.0390625, abs_max=328.3682861328125
rel_dist={7: [-326.25616409772607, 326.2561640838842]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.03515625
execution time: 1706.89 seconds
