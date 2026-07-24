## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 315.174469217
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651)
1: (-147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309)
2: (-193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485)
3: (-204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282)
4: (-188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762)
5: (-168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819)
6: (-161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108)
7: (-175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403)
8: (-213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560)
9: (-159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679)

## BASE Result
execution time: IAR + LP analysis = 1.24 + 9.22 = 10.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -315.1966706, upper bound: 315.1966706


# Binary Search by BASE starts (time budget: 2689.54 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=317.01361083984375
rel_dist={6: [-315.19656872135204, 315.19656872135204]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=317.01361083984375
rel_dist={6: [-315.19620904713327, 315.1962090416698]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=317.01361083984375
rel_dist={6: [-315.19563397098386, 315.19563397064064]}

## Binary Search Result
Binary search time: 45.52 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2644.02 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1962973, upper bound: 315.1962973
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1962973, upper bound: 315.1962973
time: 6.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.33 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.33
Output dim: 6, lower bound: -315.1962973, upper bound: 315.1962973
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.33
Output dim: 6, lower bound: -315.1962973, upper bound: 315.1962973

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1959745, upper bound: 315.1959739
time: 8.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1959739, upper bound: 315.1959745
time: 10.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1842515, upper bound: 315.1842515
time: 9.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1842515, upper bound: 315.1842515
time: 9.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.07
Output dim: 6, lower bound: -315.1959745, upper bound: 315.1959739
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.07
Output dim: 6, lower bound: -315.1959739, upper bound: 315.1959745
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.07
Output dim: 6, lower bound: -315.1842515, upper bound: 315.1842515
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.07
Output dim: 6, lower bound: -315.1842515, upper bound: 315.1842515

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1959335, upper bound: 315.1959739
time: 9.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1959745, upper bound: 315.1959334
time: 11.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1903338, upper bound: 315.1903353
time: 9.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1903338, upper bound: 315.1903353
time: 9.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1725682, upper bound: 315.1725679
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1725682, upper bound: 315.1725679
time: 9.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1744671, upper bound: 315.1744671
time: 8.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1744671, upper bound: 315.1744671
time: 8.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.19
Output dim: 6, lower bound: -315.1959335, upper bound: 315.1959739
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.19
Output dim: 6, lower bound: -315.1959745, upper bound: 315.1959334
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.19
Output dim: 6, lower bound: -315.1903338, upper bound: 315.1903353
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.19
Output dim: 6, lower bound: -315.1903338, upper bound: 315.1903353
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.19
Output dim: 6, lower bound: -315.1725682, upper bound: 315.1725679
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.19
Output dim: 6, lower bound: -315.1725682, upper bound: 315.1725679
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.19
Output dim: 6, lower bound: -315.1744671, upper bound: 315.1744671
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.19
Output dim: 6, lower bound: -315.1744671, upper bound: 315.1744671

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1959335, upper bound: 315.1959737
time: 9.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1959215, upper bound: 315.1959739
time: 10.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1914937, upper bound: 315.1914653
time: 9.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1914937, upper bound: 315.1914653
time: 10.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1834713, upper bound: 315.1834689
time: 10.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1834713, upper bound: 315.1834689
time: 8.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1886897, upper bound: 315.1886884
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1886897, upper bound: 315.1886884
time: 8.12 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.35
Output dim: 6, lower bound: -315.1959335, upper bound: 315.1959737
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.35
Output dim: 6, lower bound: -315.1959215, upper bound: 315.1959739
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.35
Output dim: 6, lower bound: -315.1914937, upper bound: 315.1914653
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.35
Output dim: 6, lower bound: -315.1914937, upper bound: 315.1914653
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.35
Output dim: 6, lower bound: -315.1834713, upper bound: 315.1834689
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.35
Output dim: 6, lower bound: -315.1834713, upper bound: 315.1834689
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.35
Output dim: 6, lower bound: -315.1886897, upper bound: 315.1886884
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.35
Output dim: 6, lower bound: -315.1886897, upper bound: 315.1886884

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1938467, upper bound: 315.1938754
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1938467, upper bound: 315.1938754
time: 8.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1715263, upper bound: 315.1715414
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1715263, upper bound: 315.1715414
time: 9.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1851065, upper bound: 315.1850312
time: 10.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1851065, upper bound: 315.1850312
time: 9.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1906180, upper bound: 315.1905925
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1906268, upper bound: 315.1905904
time: 9.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1834470, upper bound: 315.1834689
time: 9.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1834713, upper bound: 315.1834412
time: 9.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1820530, upper bound: 315.1820565
time: 9.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1820530, upper bound: 315.1820565
time: 8.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1716681, upper bound: 315.1716650
time: 12.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1716681, upper bound: 315.1716650
time: 11.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1409875, upper bound: 315.1409335
time: 10.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1409875, upper bound: 315.1409335
time: 10.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1938467, upper bound: 315.1938754
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1938467, upper bound: 315.1938754
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1715263, upper bound: 315.1715414
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1715263, upper bound: 315.1715414
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1851065, upper bound: 315.1850312
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1851065, upper bound: 315.1850312
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1906180, upper bound: 315.1905925
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1906268, upper bound: 315.1905904
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1834470, upper bound: 315.1834689
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1834713, upper bound: 315.1834412
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1820530, upper bound: 315.1820565
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1820530, upper bound: 315.1820565
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1716681, upper bound: 315.1716650
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1716681, upper bound: 315.1716650
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1409875, upper bound: 315.1409335
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 21.81
Output dim: 6, lower bound: -315.1409875, upper bound: 315.1409335

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1880856, upper bound: 315.1881105
time: 12.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1880856, upper bound: 315.1881105
time: 11.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1812581, upper bound: 315.1813355
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1812581, upper bound: 315.1813355
time: 9.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1768236, upper bound: 315.1768045
time: 11.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1768236, upper bound: 315.1768045
time: 11.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1851065, upper bound: 315.1850234
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1850987, upper bound: 315.1850312
time: 13.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1872937, upper bound: 315.1872608
time: 10.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1872937, upper bound: 315.1872608
time: 13.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1890033, upper bound: 315.1889989
time: 10.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1890346, upper bound: 315.1889753
time: 12.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1808349, upper bound: 315.1808132
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1807870, upper bound: 315.1808511
time: 7.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1817936, upper bound: 315.1817731
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1817935, upper bound: 315.1817729
time: 7.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1630074, upper bound: 315.1630077
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1630074, upper bound: 315.1630077
time: 6.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1605038, upper bound: 315.1605069
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1605038, upper bound: 315.1605069
time: 6.82 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 14.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1880856, upper bound: 315.1881105
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1880856, upper bound: 315.1881105
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1812581, upper bound: 315.1813355
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1812581, upper bound: 315.1813355
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1768236, upper bound: 315.1768045
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1768236, upper bound: 315.1768045
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1851065, upper bound: 315.1850234
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1850987, upper bound: 315.1850312
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1872937, upper bound: 315.1872608
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1872937, upper bound: 315.1872608
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1890033, upper bound: 315.1889989
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1890346, upper bound: 315.1889753
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1808349, upper bound: 315.1808132
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1807870, upper bound: 315.1808511
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1817936, upper bound: 315.1817731
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1817935, upper bound: 315.1817729
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1630074, upper bound: 315.1630077
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1630074, upper bound: 315.1630077
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1605038, upper bound: 315.1605069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 14.84
Output dim: 6, lower bound: -315.1605038, upper bound: 315.1605069

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1778815, upper bound: 315.1779871
time: 7.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1778808, upper bound: 315.1779796
time: 8.49 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 17.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 17.64
Output dim: 6, lower bound: -315.1778815, upper bound: 315.1779871
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 17.64
Output dim: 6, lower bound: -315.1778808, upper bound: 315.1779796
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1880856, upper bound: 315.1881105
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1812581, upper bound: 315.1813355
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1812581, upper bound: 315.1813355
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1768236, upper bound: 315.1768045
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1768236, upper bound: 315.1768045
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1851065, upper bound: 315.1850234
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1850987, upper bound: 315.1850312
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1872937, upper bound: 315.1872608
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1872937, upper bound: 315.1872608
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1890033, upper bound: 315.1889989
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1890346, upper bound: 315.1889753
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1808349, upper bound: 315.1808132
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1807870, upper bound: 315.1808511
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1817936, upper bound: 315.1817731
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.64
Output dim: 6, lower bound: -315.1817935, upper bound: 315.1817729
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=317.01361083984375
rel_dist={6: [-315.19656872135204, 315.19656872135204]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1714807, upper bound: 315.1714808
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1714807, upper bound: 315.1714808
time: 10.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.78 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 17.78
Output dim: 6, lower bound: -315.1714807, upper bound: 315.1714808
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 17.78
Output dim: 6, lower bound: -315.1714807, upper bound: 315.1714808
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=317.01361083984375
rel_dist={6: [-315.19620904713327, 315.1962090416698]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1949298, upper bound: 315.1949295
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1949295, upper bound: 315.1949298
time: 7.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.32
Output dim: 6, lower bound: -315.1949298, upper bound: 315.1949295
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.32
Output dim: 6, lower bound: -315.1949295, upper bound: 315.1949298

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1570188, upper bound: 315.1570217
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1570188, upper bound: 315.1570217
time: 6.39 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1864258, upper bound: 315.1864251
time: 10.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1864258, upper bound: 315.1864251
time: 10.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.87 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 21.87
Output dim: 6, lower bound: -315.1570188, upper bound: 315.1570217
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 21.87
Output dim: 6, lower bound: -315.1570188, upper bound: 315.1570217
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.87
Output dim: 6, lower bound: -315.1864258, upper bound: 315.1864251
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.87
Output dim: 6, lower bound: -315.1864258, upper bound: 315.1864251

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1837449, upper bound: 315.1837379
time: 11.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1837386, upper bound: 315.1837428
time: 7.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1792177, upper bound: 315.1792187
time: 9.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1792177, upper bound: 315.1792187
time: 7.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.24 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.24
Output dim: 6, lower bound: -315.1837449, upper bound: 315.1837379
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.24
Output dim: 6, lower bound: -315.1837386, upper bound: 315.1837428
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.24
Output dim: 6, lower bound: -315.1792177, upper bound: 315.1792187
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.24
Output dim: 6, lower bound: -315.1792177, upper bound: 315.1792187

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1837290, upper bound: 315.1837271
time: 8.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1837329, upper bound: 315.1837230
time: 7.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1821554, upper bound: 315.1821761
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1821726, upper bound: 315.1821551
time: 7.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1792177, upper bound: 315.1792185
time: 9.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1792173, upper bound: 315.1792187
time: 7.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1703168, upper bound: 315.1703163
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1703168, upper bound: 315.1703163
time: 7.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.41 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.41
Output dim: 6, lower bound: -315.1837290, upper bound: 315.1837271
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.41
Output dim: 6, lower bound: -315.1837329, upper bound: 315.1837230
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.41
Output dim: 6, lower bound: -315.1821554, upper bound: 315.1821761
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.41
Output dim: 6, lower bound: -315.1821726, upper bound: 315.1821551
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.41
Output dim: 6, lower bound: -315.1792177, upper bound: 315.1792185
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.41
Output dim: 6, lower bound: -315.1792173, upper bound: 315.1792187
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 16.41
Output dim: 6, lower bound: -315.1703168, upper bound: 315.1703163
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 16.41
Output dim: 6, lower bound: -315.1703168, upper bound: 315.1703163

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1804230, upper bound: 315.1804243
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1804291, upper bound: 315.1804184
time: 10.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1837074, upper bound: 315.1837230
time: 9.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1837329, upper bound: 315.1836913
time: 13.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1805593, upper bound: 315.1805883
time: 8.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1805664, upper bound: 315.1805710
time: 6.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1765358, upper bound: 315.1765116
time: 13.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1765358, upper bound: 315.1765116
time: 12.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1790866, upper bound: 315.1790896
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1790876, upper bound: 315.1790879
time: 14.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1302046, upper bound: 315.1302064
time: 9.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1302046, upper bound: 315.1302064
time: 6.79 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 17.56 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1804230, upper bound: 315.1804243
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1804291, upper bound: 315.1804184
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1837074, upper bound: 315.1837230
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1837329, upper bound: 315.1836913
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1805593, upper bound: 315.1805883
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1805664, upper bound: 315.1805710
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1765358, upper bound: 315.1765116
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1765358, upper bound: 315.1765116
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1790866, upper bound: 315.1790896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1790876, upper bound: 315.1790879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1302046, upper bound: 315.1302064
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.56
Output dim: 6, lower bound: -315.1302046, upper bound: 315.1302064

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1794955, upper bound: 315.1794947
time: 7.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1794955, upper bound: 315.1794947
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1314325, upper bound: 315.1314524
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1314325, upper bound: 315.1314524
time: 6.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1802350, upper bound: 315.1802331
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1802350, upper bound: 315.1802331
time: 9.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1679916, upper bound: 315.1679847
time: 10.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1679916, upper bound: 315.1679847
time: 12.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1803793, upper bound: 315.1803987
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1803793, upper bound: 315.1803987
time: 10.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1760950, upper bound: 315.1760976
time: 8.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1760950, upper bound: 315.1760976
time: 10.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1740337, upper bound: 315.1740185
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1740442, upper bound: 315.1740077
time: 8.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1737787, upper bound: 315.1737754
time: 9.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1737787, upper bound: 315.1737754
time: 9.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1765234, upper bound: 315.1765509
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1765234, upper bound: 315.1765509
time: 7.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1699131, upper bound: 315.1699030
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1699131, upper bound: 315.1699030
time: 7.77 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 16.03 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1794955, upper bound: 315.1794947
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1794955, upper bound: 315.1794947
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1314325, upper bound: 315.1314524
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1314325, upper bound: 315.1314524
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1802350, upper bound: 315.1802331
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1802350, upper bound: 315.1802331
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1679916, upper bound: 315.1679847
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1679916, upper bound: 315.1679847
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1803793, upper bound: 315.1803987
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1803793, upper bound: 315.1803987
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1760950, upper bound: 315.1760976
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1760950, upper bound: 315.1760976
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1740337, upper bound: 315.1740185
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1740442, upper bound: 315.1740077
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1737787, upper bound: 315.1737754
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1737787, upper bound: 315.1737754
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1765234, upper bound: 315.1765509
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1765234, upper bound: 315.1765509
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1699131, upper bound: 315.1699030
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 16.03
Output dim: 6, lower bound: -315.1699131, upper bound: 315.1699030

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1761592, upper bound: 315.1761650
time: 9.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1761592, upper bound: 315.1761650
time: 9.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1794906, upper bound: 315.1794947
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1794955, upper bound: 315.1794847
time: 7.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1778984, upper bound: 315.1778804
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1778984, upper bound: 315.1778804
time: 7.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1767098, upper bound: 315.1766935
time: 8.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1767098, upper bound: 315.1766935
time: 7.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1698570, upper bound: 315.1698940
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1698387, upper bound: 315.1699096
time: 7.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1766057, upper bound: 315.1766413
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1766057, upper bound: 315.1766413
time: 7.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -175.4864807, 140.6464844, -175.4864807, 140.6464844, -316.1329651, 316.1329651
1: -147.6187286, 123.9792175, -147.6187286, 123.9792175, -271.5979309, 271.5979309
2: -193.5413055, 126.3884354, -193.5413055, 126.3884354, -319.9297485, 319.9297485
3: -204.7216034, 108.3048859, -204.7216034, 108.3048859, -313.0264282, 313.0264282
4: -188.8656311, 144.3499298, -188.8656311, 144.3499298, -333.2155762, 333.2155762
5: -168.9125214, 131.7426605, -168.9125214, 131.7426605, -300.6551819, 300.6551819
6: -161.5124664, 155.5011597, -161.5124664, 155.5011597, -317.0136108, 317.0136108
7: -175.6107178, 147.9506073, -175.6107178, 147.9506073, -323.5613403, 323.5613403
8: -213.8157196, 147.4116516, -213.8157196, 147.4116516, -361.2273560, 361.2273560
9: -159.9953308, 158.1130371, -159.9953308, 158.1130371, -318.1083679, 318.1083679

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1731246, upper bound: 315.1731066
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1731246, upper bound: 315.1731066
time: 7.82 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 16.63 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1761592, upper bound: 315.1761650
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1761592, upper bound: 315.1761650
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1794906, upper bound: 315.1794947
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1794955, upper bound: 315.1794847
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1778984, upper bound: 315.1778804
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1778984, upper bound: 315.1778804
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1767098, upper bound: 315.1766935
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1767098, upper bound: 315.1766935
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1698570, upper bound: 315.1698940
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1698387, upper bound: 315.1699096
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1766057, upper bound: 315.1766413
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1766057, upper bound: 315.1766413
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1731246, upper bound: 315.1731066
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.63
Output dim: 6, lower bound: -315.1731246, upper bound: 315.1731066
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.63
Output dim: 6, lower bound: -315.1760950, upper bound: 315.1760976
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 16.63
Output dim: 6, lower bound: -315.1765234, upper bound: 315.1765509
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 16.63
Output dim: 6, lower bound: -315.1765234, upper bound: 315.1765509
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=317.01361083984375
rel_dist={6: [-315.1963780228785, 315.19637802287843]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1246.18 seconds
