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
execution time: IAR + LP analysis = 1.22 + 11.05 = 12.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -326.2561776, upper bound: 326.2561776


# Binary Search by BASE starts (time budget: 2687.74 seconds, max iter: 100)

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
Binary search time: 45.04 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2642.70 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2168196, upper bound: 326.2168196
time: 9.01 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2168196, upper bound: 326.2168196
time: 9.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.04
Output dim: 7, lower bound: -326.2168196, upper bound: 326.2168196
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.04
Output dim: 7, lower bound: -326.2168196, upper bound: 326.2168196

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2168196, upper bound: 326.2168128
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2168128, upper bound: 326.2168196
time: 8.73 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1936805, upper bound: 326.1936805
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1936805, upper bound: 326.1936805
time: 7.22 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.69 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.69
Output dim: 7, lower bound: -326.2168196, upper bound: 326.2168128
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.69
Output dim: 7, lower bound: -326.2168128, upper bound: 326.2168196
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.69
Output dim: 7, lower bound: -326.1936805, upper bound: 326.1936805
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.69
Output dim: 7, lower bound: -326.1936805, upper bound: 326.1936805

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
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2123436, upper bound: 326.2123401
time: 9.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2123436, upper bound: 326.2123401
time: 7.29 seconds

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
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1568507, upper bound: 326.1568731
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1568507, upper bound: 326.1568731
time: 5.93 seconds

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1934506, upper bound: 326.1934371
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1934371, upper bound: 326.1934506
time: 6.29 seconds

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
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1922348, upper bound: 326.1922348
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1922348, upper bound: 326.1922348
time: 7.60 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.68 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.68
Output dim: 7, lower bound: -326.2123436, upper bound: 326.2123401
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.68
Output dim: 7, lower bound: -326.2123436, upper bound: 326.2123401
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.68
Output dim: 7, lower bound: -326.1568507, upper bound: 326.1568731
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.68
Output dim: 7, lower bound: -326.1568507, upper bound: 326.1568731
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.68
Output dim: 7, lower bound: -326.1934506, upper bound: 326.1934371
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.68
Output dim: 7, lower bound: -326.1934371, upper bound: 326.1934506
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.68
Output dim: 7, lower bound: -326.1922348, upper bound: 326.1922348
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.68
Output dim: 7, lower bound: -326.1922348, upper bound: 326.1922348

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1814733, upper bound: 326.1814719
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1814733, upper bound: 326.1814719
time: 6.63 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1956857, upper bound: 326.1956842
time: 8.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1956857, upper bound: 326.1956842
time: 8.64 seconds

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
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1821909, upper bound: 326.1821845
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1821909, upper bound: 326.1821845
time: 5.87 seconds

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
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1922802, upper bound: 326.1922995
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1922779, upper bound: 326.1922995
time: 7.60 seconds

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
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1797023, upper bound: 326.1797023
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1797023, upper bound: 326.1797023
time: 6.89 seconds

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1922335, upper bound: 326.1922348
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1922348, upper bound: 326.1922335
time: 8.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1814733, upper bound: 326.1814719
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1814733, upper bound: 326.1814719
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1956857, upper bound: 326.1956842
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1956857, upper bound: 326.1956842
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1821909, upper bound: 326.1821845
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1821909, upper bound: 326.1821845
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1922802, upper bound: 326.1922995
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1922779, upper bound: 326.1922995
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1797023, upper bound: 326.1797023
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1797023, upper bound: 326.1797023
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1922335, upper bound: 326.1922348
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.75
Output dim: 7, lower bound: -326.1922348, upper bound: 326.1922335

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1814733, upper bound: 326.1814592
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1814595, upper bound: 326.1814719
time: 5.29 seconds

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1771724, upper bound: 326.1771683
time: 6.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1771724, upper bound: 326.1771683
time: 6.82 seconds

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
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1843847, upper bound: 326.1843535
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1843847, upper bound: 326.1843535
time: 6.24 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 224

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1761825, upper bound: 326.1761627
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1761825, upper bound: 326.1761627
time: 5.90 seconds

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1821909, upper bound: 326.1821739
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1821881, upper bound: 326.1821845
time: 6.56 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1670345, upper bound: 326.1669912
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1670345, upper bound: 326.1669912
time: 6.94 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1784953, upper bound: 326.1785078
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1784953, upper bound: 326.1785078
time: 6.89 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1813685, upper bound: 326.1813730
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1813685, upper bound: 326.1813730
time: 5.67 seconds

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1796966, upper bound: 326.1797023
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1797023, upper bound: 326.1796966
time: 5.46 seconds

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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1656195, upper bound: 326.1656195
time: 8.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1656195, upper bound: 326.1656195
time: 7.19 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1922333, upper bound: 326.1922348
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1922335, upper bound: 326.1922347
time: 6.92 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1922348, upper bound: 326.1922335
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1922344, upper bound: 326.1922335
time: 7.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1814733, upper bound: 326.1814592
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1814595, upper bound: 326.1814719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1771724, upper bound: 326.1771683
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1771724, upper bound: 326.1771683
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1843847, upper bound: 326.1843535
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1843847, upper bound: 326.1843535
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1761825, upper bound: 326.1761627
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1761825, upper bound: 326.1761627
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1821909, upper bound: 326.1821739
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1821881, upper bound: 326.1821845
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1670345, upper bound: 326.1669912
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1670345, upper bound: 326.1669912
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1784953, upper bound: 326.1785078
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1784953, upper bound: 326.1785078
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1813685, upper bound: 326.1813730
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1813685, upper bound: 326.1813730
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1796966, upper bound: 326.1797023
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1797023, upper bound: 326.1796966
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1656195, upper bound: 326.1656195
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1656195, upper bound: 326.1656195
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1922333, upper bound: 326.1922348
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1922335, upper bound: 326.1922347
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1922348, upper bound: 326.1922335
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.26
Output dim: 7, lower bound: -326.1922344, upper bound: 326.1922335

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1810440, upper bound: 326.1810355
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1810440, upper bound: 326.1810355
time: 6.02 seconds

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1583176, upper bound: 326.1583144
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1583176, upper bound: 326.1583144
time: 7.01 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 224

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1699729, upper bound: 326.1699638
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1699729, upper bound: 326.1699638
time: 6.59 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1771724, upper bound: 326.1771669
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1771694, upper bound: 326.1771683
time: 6.49 seconds

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
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1843847, upper bound: 326.1843535
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1843847, upper bound: 326.1843530
time: 6.86 seconds

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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1843762, upper bound: 326.1843537
time: 9.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1843847, upper bound: 326.1843507
time: 7.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1280579, upper bound: 326.1280407
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1280579, upper bound: 326.1280407
time: 6.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1457938, upper bound: 326.1457810
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1457938, upper bound: 326.1457810
time: 5.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1660320, upper bound: 326.1660357
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1660320, upper bound: 326.1660357
time: 8.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1818745, upper bound: 326.1818810
time: 7.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1818732, upper bound: 326.1818822
time: 6.25 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 16.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1810440, upper bound: 326.1810355
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1810440, upper bound: 326.1810355
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1583176, upper bound: 326.1583144
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1583176, upper bound: 326.1583144
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1699729, upper bound: 326.1699638
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1699729, upper bound: 326.1699638
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1771724, upper bound: 326.1771669
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1771694, upper bound: 326.1771683
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1843847, upper bound: 326.1843535
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1843847, upper bound: 326.1843530
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1843762, upper bound: 326.1843537
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1843847, upper bound: 326.1843507
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1280579, upper bound: 326.1280407
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1280579, upper bound: 326.1280407
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1457938, upper bound: 326.1457810
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1457938, upper bound: 326.1457810
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1660320, upper bound: 326.1660357
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1660320, upper bound: 326.1660357
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1818745, upper bound: 326.1818810
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.71
Output dim: 7, lower bound: -326.1818732, upper bound: 326.1818822
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.71
Output dim: 7, lower bound: -326.1784953, upper bound: 326.1785078
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.71
Output dim: 7, lower bound: -326.1784953, upper bound: 326.1785078
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.71
Output dim: 7, lower bound: -326.1813685, upper bound: 326.1813730
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.71
Output dim: 7, lower bound: -326.1813685, upper bound: 326.1813730
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.71
Output dim: 7, lower bound: -326.1796966, upper bound: 326.1797023
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.71
Output dim: 7, lower bound: -326.1797023, upper bound: 326.1796966
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.71
Output dim: 7, lower bound: -326.1922333, upper bound: 326.1922348
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.71
Output dim: 7, lower bound: -326.1922335, upper bound: 326.1922347
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.71
Output dim: 7, lower bound: -326.1922348, upper bound: 326.1922335
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.71
Output dim: 7, lower bound: -326.1922344, upper bound: 326.1922335
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=328.3682861328125
rel_dist={7: [-326.25613672106726, 326.2561367077651]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2446666, upper bound: 326.2446666
time: 8.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2446666, upper bound: 326.2446666
time: 7.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.84 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.84
Output dim: 7, lower bound: -326.2446666, upper bound: 326.2446666
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.84
Output dim: 7, lower bound: -326.2446666, upper bound: 326.2446666

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2401531, upper bound: 326.2401531
time: 9.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2401531, upper bound: 326.2401531
time: 7.31 seconds

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
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2228441, upper bound: 326.2228441
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2228441, upper bound: 326.2228441
time: 8.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.72 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.72
Output dim: 7, lower bound: -326.2401531, upper bound: 326.2401531
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.72
Output dim: 7, lower bound: -326.2401531, upper bound: 326.2401531
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.72
Output dim: 7, lower bound: -326.2228441, upper bound: 326.2228441
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.72
Output dim: 7, lower bound: -326.2228441, upper bound: 326.2228441

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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2321368, upper bound: 326.2321368
time: 7.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2321368, upper bound: 326.2321368
time: 7.80 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2401526, upper bound: 326.2401531
time: 9.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2401531, upper bound: 326.2401526
time: 8.70 seconds

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
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2045295, upper bound: 326.2045295
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2045295, upper bound: 326.2045295
time: 7.85 seconds

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
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1979086, upper bound: 326.1979086
time: 9.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1979086, upper bound: 326.1979086
time: 9.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.86
Output dim: 7, lower bound: -326.2321368, upper bound: 326.2321368
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.86
Output dim: 7, lower bound: -326.2321368, upper bound: 326.2321368
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.86
Output dim: 7, lower bound: -326.2401526, upper bound: 326.2401531
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.86
Output dim: 7, lower bound: -326.2401531, upper bound: 326.2401526
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.86
Output dim: 7, lower bound: -326.2045295, upper bound: 326.2045295
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.86
Output dim: 7, lower bound: -326.2045295, upper bound: 326.2045295
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.86
Output dim: 7, lower bound: -326.1979086, upper bound: 326.1979086
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.86
Output dim: 7, lower bound: -326.1979086, upper bound: 326.1979086

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2320173, upper bound: 326.2320170
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2320170, upper bound: 326.2320173
time: 10.06 seconds

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2321362, upper bound: 326.2321368
time: 9.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2321368, upper bound: 326.2321362
time: 8.15 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2336442, upper bound: 326.2336441
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2336442, upper bound: 326.2336441
time: 7.53 seconds

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
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2220853, upper bound: 326.2220871
time: 8.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2220853, upper bound: 326.2220871
time: 8.39 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1945709, upper bound: 326.1945709
time: 8.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1945709, upper bound: 326.1945709
time: 7.90 seconds

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
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2045287, upper bound: 326.2045295
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2045295, upper bound: 326.2045287
time: 8.49 seconds

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
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1939321, upper bound: 326.1939322
time: 9.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1939322, upper bound: 326.1939321
time: 8.30 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 224

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1278978, upper bound: 326.1278978
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1278978, upper bound: 326.1278978
time: 7.20 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.2320173, upper bound: 326.2320170
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.2320170, upper bound: 326.2320173
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.2321362, upper bound: 326.2321368
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.2321368, upper bound: 326.2321362
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.2336442, upper bound: 326.2336441
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.2336442, upper bound: 326.2336441
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.2220853, upper bound: 326.2220871
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.2220853, upper bound: 326.2220871
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.1945709, upper bound: 326.1945709
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.1945709, upper bound: 326.1945709
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.2045287, upper bound: 326.2045295
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.2045295, upper bound: 326.2045287
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.1939321, upper bound: 326.1939322
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.1939322, upper bound: 326.1939321
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.1278978, upper bound: 326.1278978
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.36
Output dim: 7, lower bound: -326.1278978, upper bound: 326.1278978

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2149481, upper bound: 326.2149518
time: 9.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2149481, upper bound: 326.2149518
time: 8.99 seconds

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
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1944274, upper bound: 326.1944228
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1944274, upper bound: 326.1944228
time: 7.66 seconds

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
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1387414, upper bound: 326.1387428
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1387414, upper bound: 326.1387428
time: 6.15 seconds

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1667155, upper bound: 326.1667116
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1667155, upper bound: 326.1667116
time: 6.87 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1861750, upper bound: 326.1861770
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1861750, upper bound: 326.1861770
time: 7.23 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1599431, upper bound: 326.1599428
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1599431, upper bound: 326.1599428
time: 6.62 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2117300, upper bound: 326.2117319
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2117300, upper bound: 326.2117319
time: 7.63 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2220853, upper bound: 326.2220833
time: 8.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2220815, upper bound: 326.2220871
time: 10.13 seconds

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1945671, upper bound: 326.1945709
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1945709, upper bound: 326.1945671
time: 7.83 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1398142, upper bound: 326.1398142
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1398142, upper bound: 326.1398142
time: 8.49 seconds

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
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2031224, upper bound: 326.2031176
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2031157, upper bound: 326.2031236
time: 6.82 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1770596, upper bound: 326.1770592
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1770596, upper bound: 326.1770592
time: 7.47 seconds

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1938713, upper bound: 326.1938754
time: 7.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1938755, upper bound: 326.1938711
time: 6.95 seconds

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
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1939322, upper bound: 326.1939321
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1939322, upper bound: 326.1939321
time: 7.45 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.2149481, upper bound: 326.2149518
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.2149481, upper bound: 326.2149518
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1944274, upper bound: 326.1944228
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1944274, upper bound: 326.1944228
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1387414, upper bound: 326.1387428
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1387414, upper bound: 326.1387428
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1667155, upper bound: 326.1667116
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1667155, upper bound: 326.1667116
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1861750, upper bound: 326.1861770
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1861750, upper bound: 326.1861770
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1599431, upper bound: 326.1599428
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1599431, upper bound: 326.1599428
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.2117300, upper bound: 326.2117319
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.2117300, upper bound: 326.2117319
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.2220853, upper bound: 326.2220833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.2220815, upper bound: 326.2220871
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1945671, upper bound: 326.1945709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1945709, upper bound: 326.1945671
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1398142, upper bound: 326.1398142
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1398142, upper bound: 326.1398142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.2031224, upper bound: 326.2031176
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.2031157, upper bound: 326.2031236
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1770596, upper bound: 326.1770592
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1770596, upper bound: 326.1770592
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1938713, upper bound: 326.1938754
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1938755, upper bound: 326.1938711
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1939322, upper bound: 326.1939321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.50
Output dim: 7, lower bound: -326.1939322, upper bound: 326.1939321

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1716379, upper bound: 326.1716535
time: 7.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1716379, upper bound: 326.1716535
time: 7.74 seconds

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2149481, upper bound: 326.2149471
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2149423, upper bound: 326.2149518
time: 9.75 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1815288, upper bound: 326.1815263
time: 8.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1815288, upper bound: 326.1815263
time: 8.26 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1567160, upper bound: 326.1567168
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1567160, upper bound: 326.1567168
time: 6.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1860918, upper bound: 326.1860930
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1860919, upper bound: 326.1860939
time: 7.76 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 15.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.80
Output dim: 7, lower bound: -326.1716379, upper bound: 326.1716535
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.80
Output dim: 7, lower bound: -326.1716379, upper bound: 326.1716535
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.80
Output dim: 7, lower bound: -326.2149481, upper bound: 326.2149471
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.80
Output dim: 7, lower bound: -326.2149423, upper bound: 326.2149518
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.80
Output dim: 7, lower bound: -326.1815288, upper bound: 326.1815263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.80
Output dim: 7, lower bound: -326.1815288, upper bound: 326.1815263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.80
Output dim: 7, lower bound: -326.1567160, upper bound: 326.1567168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.80
Output dim: 7, lower bound: -326.1567160, upper bound: 326.1567168
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.80
Output dim: 7, lower bound: -326.1860918, upper bound: 326.1860930
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.80
Output dim: 7, lower bound: -326.1860919, upper bound: 326.1860939
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.1861750, upper bound: 326.1861770
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.2117300, upper bound: 326.2117319
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.2117300, upper bound: 326.2117319
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.2220853, upper bound: 326.2220833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.2220815, upper bound: 326.2220871
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.1945671, upper bound: 326.1945709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.1945709, upper bound: 326.1945671
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.2031224, upper bound: 326.2031176
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.2031157, upper bound: 326.2031236
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.1770596, upper bound: 326.1770592
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.1770596, upper bound: 326.1770592
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.1938713, upper bound: 326.1938754
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.1938755, upper bound: 326.1938711
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.1939322, upper bound: 326.1939321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.80
Output dim: 7, lower bound: -326.1939322, upper bound: 326.1939321
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=328.3682861328125
rel_dist={7: [-326.2560128858547, 326.2560128858547]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2443533, upper bound: 326.2443534
time: 9.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2443533, upper bound: 326.2443534
time: 9.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.96 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.96
Output dim: 7, lower bound: -326.2443533, upper bound: 326.2443534
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.96
Output dim: 7, lower bound: -326.2443533, upper bound: 326.2443534

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1996551, upper bound: 326.1996551
time: 9.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1996551, upper bound: 326.1996551
time: 9.60 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2436826, upper bound: 326.2436847
time: 9.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2436826, upper bound: 326.2436826
time: 10.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.57
Output dim: 7, lower bound: -326.1996551, upper bound: 326.1996551
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.57
Output dim: 7, lower bound: -326.1996551, upper bound: 326.1996551
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.57
Output dim: 7, lower bound: -326.2436826, upper bound: 326.2436847
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.57
Output dim: 7, lower bound: -326.2436826, upper bound: 326.2436826

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764195, upper bound: 326.1764195
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764195, upper bound: 326.1764195
time: 8.02 seconds

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
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1331560, upper bound: 326.1331560
time: 7.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1331560, upper bound: 326.1331560
time: 7.65 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2436826, upper bound: 326.2436840
time: 11.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2436812, upper bound: 326.2436847
time: 11.02 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2077135, upper bound: 326.2077135
time: 9.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2077135, upper bound: 326.2077135
time: 9.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.61 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.61
Output dim: 7, lower bound: -326.1764195, upper bound: 326.1764195
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.61
Output dim: 7, lower bound: -326.1764195, upper bound: 326.1764195
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.61
Output dim: 7, lower bound: -326.1331560, upper bound: 326.1331560
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.61
Output dim: 7, lower bound: -326.1331560, upper bound: 326.1331560
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.61
Output dim: 7, lower bound: -326.2436826, upper bound: 326.2436840
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.61
Output dim: 7, lower bound: -326.2436812, upper bound: 326.2436847
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.61
Output dim: 7, lower bound: -326.2077135, upper bound: 326.2077135
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.61
Output dim: 7, lower bound: -326.2077135, upper bound: 326.2077135

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
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1697067, upper bound: 326.1697037
time: 8.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1697037, upper bound: 326.1697067
time: 10.23 seconds

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764195, upper bound: 326.1764194
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764194, upper bound: 326.1764195
time: 8.55 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1689048, upper bound: 326.1689012
time: 8.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1689048, upper bound: 326.1689012
time: 8.27 seconds

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
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2156257, upper bound: 326.2156279
time: 8.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2156257, upper bound: 326.2156279
time: 9.05 seconds

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
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2077135, upper bound: 326.2077065
time: 9.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2077065, upper bound: 326.2077135
time: 9.68 seconds

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
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1710484, upper bound: 326.1710485
time: 9.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1710484, upper bound: 326.1710485
time: 9.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.1697067, upper bound: 326.1697037
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.1697037, upper bound: 326.1697067
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.1764195, upper bound: 326.1764194
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.1764194, upper bound: 326.1764195
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.1689048, upper bound: 326.1689012
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.1689048, upper bound: 326.1689012
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.2156257, upper bound: 326.2156279
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.2156257, upper bound: 326.2156279
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.2077135, upper bound: 326.2077065
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.2077065, upper bound: 326.2077135
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.1710484, upper bound: 326.1710485
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 20.92
Output dim: 7, lower bound: -326.1710484, upper bound: 326.1710485

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
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 224

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764195, upper bound: 326.1764193
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764194, upper bound: 326.1764194
time: 8.76 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 224

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764194, upper bound: 326.1764194
time: 7.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764193, upper bound: 326.1764195
time: 8.06 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2025974, upper bound: 326.2025995
time: 9.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2025974, upper bound: 326.2025995
time: 10.02 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1966699, upper bound: 326.1966738
time: 8.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1966699, upper bound: 326.1966738
time: 8.81 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1576880, upper bound: 326.1576880
time: 7.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1576880, upper bound: 326.1576880
time: 7.65 seconds

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1781251, upper bound: 326.1781269
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1781251, upper bound: 326.1781269
time: 7.24 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 15.63 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.1764195, upper bound: 326.1764193
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.1764194, upper bound: 326.1764194
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.1764194, upper bound: 326.1764194
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.1764193, upper bound: 326.1764195
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.2025974, upper bound: 326.2025995
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.2025974, upper bound: 326.2025995
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.1966699, upper bound: 326.1966738
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.1966699, upper bound: 326.1966738
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.1576880, upper bound: 326.1576880
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.1576880, upper bound: 326.1576880
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.1781251, upper bound: 326.1781269
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 7, lower bound: -326.1781251, upper bound: 326.1781269

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1227648, upper bound: 326.1227639
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1227648, upper bound: 326.1227639
time: 8.17 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1530715, upper bound: 326.1530704
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1530715, upper bound: 326.1530704
time: 7.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1760765, upper bound: 326.1760813
time: 8.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1760795, upper bound: 326.1760784
time: 10.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764193, upper bound: 326.1764178
time: 9.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1764169, upper bound: 326.1764195
time: 8.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1816585, upper bound: 326.1816607
time: 8.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1816585, upper bound: 326.1816607
time: 8.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1589748, upper bound: 326.1589757
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1589748, upper bound: 326.1589757
time: 6.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1492470, upper bound: 326.1492480
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1492470, upper bound: 326.1492480
time: 7.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1966699, upper bound: 326.1966734
time: 9.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1966697, upper bound: 326.1966738
time: 9.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1775422, upper bound: 326.1775449
time: 8.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1775417, upper bound: 326.1775453
time: 6.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1781240, upper bound: 326.1781246
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1781220, upper bound: 326.1781264
time: 7.65 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.00 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1227648, upper bound: 326.1227639
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1227648, upper bound: 326.1227639
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1530715, upper bound: 326.1530704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1530715, upper bound: 326.1530704
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1760765, upper bound: 326.1760813
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1760795, upper bound: 326.1760784
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1764193, upper bound: 326.1764178
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1764169, upper bound: 326.1764195
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1816585, upper bound: 326.1816607
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1816585, upper bound: 326.1816607
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1589748, upper bound: 326.1589757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1589748, upper bound: 326.1589757
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1492470, upper bound: 326.1492480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1492470, upper bound: 326.1492480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1966699, upper bound: 326.1966734
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1966697, upper bound: 326.1966738
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1775422, upper bound: 326.1775449
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1775417, upper bound: 326.1775453
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1781240, upper bound: 326.1781246
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.00
Output dim: 7, lower bound: -326.1781220, upper bound: 326.1781264

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1760751, upper bound: 326.1760813
time: 10.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1760762, upper bound: 326.1760804
time: 10.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1756417, upper bound: 326.1756389
time: 7.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1756409, upper bound: 326.1756380
time: 8.13 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 20.03 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 20.03
Output dim: 7, lower bound: -326.1760751, upper bound: 326.1760813
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 20.03
Output dim: 7, lower bound: -326.1760762, upper bound: 326.1760804
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 20.03
Output dim: 7, lower bound: -326.1756417, upper bound: 326.1756389
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 20.03
Output dim: 7, lower bound: -326.1756409, upper bound: 326.1756380
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.03
Output dim: 7, lower bound: -326.1764193, upper bound: 326.1764178
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.03
Output dim: 7, lower bound: -326.1764169, upper bound: 326.1764195
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.03
Output dim: 7, lower bound: -326.1816585, upper bound: 326.1816607
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.03
Output dim: 7, lower bound: -326.1816585, upper bound: 326.1816607
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.03
Output dim: 7, lower bound: -326.1966699, upper bound: 326.1966734
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.03
Output dim: 7, lower bound: -326.1966697, upper bound: 326.1966738
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.03
Output dim: 7, lower bound: -326.1775422, upper bound: 326.1775449
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.03
Output dim: 7, lower bound: -326.1775417, upper bound: 326.1775453
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.03
Output dim: 7, lower bound: -326.1781240, upper bound: 326.1781246
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.03
Output dim: 7, lower bound: -326.1781220, upper bound: 326.1781264
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=328.3682861328125
rel_dist={7: [-326.25584232239004, 326.2558422835341]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1821.21 seconds
