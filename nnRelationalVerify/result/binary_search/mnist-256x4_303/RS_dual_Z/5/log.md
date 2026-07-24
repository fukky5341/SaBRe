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
execution time: IAR + LP analysis = 1.25 + 9.20 = 10.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -315.1966706, upper bound: 315.1966706


# Binary Search by BASE starts (time budget: 2689.55 seconds, max iter: 100)

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
Binary search time: 42.84 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2646.71 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1965007, upper bound: 315.1965687
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1965687, upper bound: 315.1965007
time: 6.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.15
Output dim: 6, lower bound: -315.1965007, upper bound: 315.1965687
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.15
Output dim: 6, lower bound: -315.1965687, upper bound: 315.1965007

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1921850, upper bound: 315.1922030
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1921850, upper bound: 315.1922030
time: 7.34 seconds

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
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1922030, upper bound: 315.1921850
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1922030, upper bound: 315.1921850
time: 7.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.09
Output dim: 6, lower bound: -315.1921850, upper bound: 315.1922030
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.09
Output dim: 6, lower bound: -315.1921850, upper bound: 315.1922030
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.09
Output dim: 6, lower bound: -315.1922030, upper bound: 315.1921850
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.09
Output dim: 6, lower bound: -315.1922030, upper bound: 315.1921850

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900011, upper bound: 315.1900368
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900011, upper bound: 315.1900368
time: 8.03 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900011, upper bound: 315.1900368
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900011, upper bound: 315.1900368
time: 8.04 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900368, upper bound: 315.1900011
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900368, upper bound: 315.1900011
time: 6.91 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900368, upper bound: 315.1900011
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900368, upper bound: 315.1900011
time: 7.14 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.24
Output dim: 6, lower bound: -315.1900011, upper bound: 315.1900368
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.24
Output dim: 6, lower bound: -315.1900011, upper bound: 315.1900368
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.24
Output dim: 6, lower bound: -315.1900011, upper bound: 315.1900368
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.24
Output dim: 6, lower bound: -315.1900011, upper bound: 315.1900368
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.24
Output dim: 6, lower bound: -315.1900368, upper bound: 315.1900011
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.24
Output dim: 6, lower bound: -315.1900368, upper bound: 315.1900011
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.24
Output dim: 6, lower bound: -315.1900368, upper bound: 315.1900011
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.24
Output dim: 6, lower bound: -315.1900368, upper bound: 315.1900011

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1808951, upper bound: 315.1809013
time: 7.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1808913, upper bound: 315.1809013
time: 8.08 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1808951, upper bound: 315.1809013
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1808913, upper bound: 315.1809013
time: 7.97 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1808951, upper bound: 315.1809013
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1808913, upper bound: 315.1809013
time: 7.40 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1808951, upper bound: 315.1809013
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1808913, upper bound: 315.1809013
time: 7.64 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808913
time: 9.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808951
time: 6.65 seconds

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808913
time: 8.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808951
time: 6.29 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808913
time: 9.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808951
time: 6.35 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808913
time: 9.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808951
time: 6.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1808951, upper bound: 315.1809013
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1808913, upper bound: 315.1809013
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1808951, upper bound: 315.1809013
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1808913, upper bound: 315.1809013
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1808951, upper bound: 315.1809013
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1808913, upper bound: 315.1809013
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1808951, upper bound: 315.1809013
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1808913, upper bound: 315.1809013
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808913
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808951
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808913
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808951
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808913
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808951
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808913
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.03
Output dim: 6, lower bound: -315.1809013, upper bound: 315.1808951

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269309
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269309
time: 6.14 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269327, upper bound: 315.1269436
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269327, upper bound: 315.1269436
time: 6.15 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269309
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269309
time: 6.38 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269327, upper bound: 315.1269436
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269327, upper bound: 315.1269436
time: 6.23 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269349, upper bound: 315.1269344
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269349, upper bound: 315.1269344
time: 6.05 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269257, upper bound: 315.1269603
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269257, upper bound: 315.1269603
time: 6.22 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269349, upper bound: 315.1269344
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269349, upper bound: 315.1269344
time: 6.04 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269257, upper bound: 315.1269603
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269257, upper bound: 315.1269603
time: 5.83 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269258
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269258
time: 6.34 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269344, upper bound: 315.1269349
time: 7.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269344, upper bound: 315.1269349
time: 7.16 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269258
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269258
time: 6.58 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269344, upper bound: 315.1269349
time: 8.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269344, upper bound: 315.1269349
time: 8.16 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269436, upper bound: 315.1269327
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269436, upper bound: 315.1269327
time: 6.12 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269309, upper bound: 315.1269603
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269309, upper bound: 315.1269603
time: 6.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269436, upper bound: 315.1269327
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269436, upper bound: 315.1269327
time: 6.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269309, upper bound: 315.1269603
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269309, upper bound: 315.1269603
time: 6.19 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269309
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269309
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269327, upper bound: 315.1269436
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269327, upper bound: 315.1269436
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269309
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269309
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269327, upper bound: 315.1269436
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269327, upper bound: 315.1269436
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269349, upper bound: 315.1269344
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269349, upper bound: 315.1269344
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269257, upper bound: 315.1269603
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269257, upper bound: 315.1269603
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269349, upper bound: 315.1269344
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269349, upper bound: 315.1269344
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269257, upper bound: 315.1269603
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269257, upper bound: 315.1269603
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269258
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269258
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269344, upper bound: 315.1269349
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269344, upper bound: 315.1269349
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269258
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269603, upper bound: 315.1269258
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269344, upper bound: 315.1269349
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269344, upper bound: 315.1269349
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269436, upper bound: 315.1269327
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269436, upper bound: 315.1269327
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269309, upper bound: 315.1269603
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269309, upper bound: 315.1269603
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269436, upper bound: 315.1269327
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269436, upper bound: 315.1269327
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269309, upper bound: 315.1269603
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.70
Output dim: 6, lower bound: -315.1269309, upper bound: 315.1269603
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=317.01361083984375
rel_dist={6: [-315.19656872135204, 315.19656872135204]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1965322, upper bound: 315.1966217
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1966217, upper bound: 315.1965322
time: 6.13 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.52
Output dim: 6, lower bound: -315.1965322, upper bound: 315.1966217
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.52
Output dim: 6, lower bound: -315.1966217, upper bound: 315.1965322

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1922537, upper bound: 315.1922774
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1922537, upper bound: 315.1922774
time: 6.85 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1922774, upper bound: 315.1922538
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1922774, upper bound: 315.1922538
time: 6.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.75 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.75
Output dim: 6, lower bound: -315.1922537, upper bound: 315.1922774
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.75
Output dim: 6, lower bound: -315.1922537, upper bound: 315.1922774
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.75
Output dim: 6, lower bound: -315.1922774, upper bound: 315.1922538
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.75
Output dim: 6, lower bound: -315.1922774, upper bound: 315.1922538

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900758, upper bound: 315.1901227
time: 7.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900758, upper bound: 315.1901226
time: 6.82 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900758, upper bound: 315.1901227
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1900758, upper bound: 315.1901227
time: 7.22 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901226, upper bound: 315.1900758
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901226, upper bound: 315.1900758
time: 7.18 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901226, upper bound: 315.1900758
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901226, upper bound: 315.1900758
time: 6.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.97
Output dim: 6, lower bound: -315.1900758, upper bound: 315.1901227
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.97
Output dim: 6, lower bound: -315.1900758, upper bound: 315.1901226
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.97
Output dim: 6, lower bound: -315.1900758, upper bound: 315.1901227
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.97
Output dim: 6, lower bound: -315.1900758, upper bound: 315.1901227
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.97
Output dim: 6, lower bound: -315.1901226, upper bound: 315.1900758
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.97
Output dim: 6, lower bound: -315.1901226, upper bound: 315.1900758
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.97
Output dim: 6, lower bound: -315.1901226, upper bound: 315.1900758
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.97
Output dim: 6, lower bound: -315.1901226, upper bound: 315.1900758

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809317, upper bound: 315.1809411
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809271, upper bound: 315.1809413
time: 7.76 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809317, upper bound: 315.1809411
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809271, upper bound: 315.1809413
time: 7.12 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809317, upper bound: 315.1809411
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809271, upper bound: 315.1809413
time: 7.17 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809317, upper bound: 315.1809411
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809271, upper bound: 315.1809413
time: 6.75 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809413, upper bound: 315.1809271
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809411, upper bound: 315.1809317
time: 8.02 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809413, upper bound: 315.1809271
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809411, upper bound: 315.1809317
time: 8.13 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809413, upper bound: 315.1809271
time: 8.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809411, upper bound: 315.1809317
time: 7.80 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809413, upper bound: 315.1809271
time: 9.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809411, upper bound: 315.1809317
time: 7.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809317, upper bound: 315.1809411
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809271, upper bound: 315.1809413
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809317, upper bound: 315.1809411
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809271, upper bound: 315.1809413
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809317, upper bound: 315.1809411
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809271, upper bound: 315.1809413
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809317, upper bound: 315.1809411
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809271, upper bound: 315.1809413
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809413, upper bound: 315.1809271
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809411, upper bound: 315.1809317
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809413, upper bound: 315.1809271
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809411, upper bound: 315.1809317
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809413, upper bound: 315.1809271
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809411, upper bound: 315.1809317
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809413, upper bound: 315.1809271
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 6, lower bound: -315.1809411, upper bound: 315.1809317

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270286, upper bound: 315.1269830
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270286, upper bound: 315.1269830
time: 6.62 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269867, upper bound: 315.1270031
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269867, upper bound: 315.1270031
time: 5.42 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270286, upper bound: 315.1269830
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270286, upper bound: 315.1269830
time: 6.72 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269867, upper bound: 315.1270031
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269867, upper bound: 315.1270031
time: 5.82 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269911, upper bound: 315.1269882
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269911, upper bound: 315.1269882
time: 6.85 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269773, upper bound: 315.1270284
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269773, upper bound: 315.1270284
time: 6.99 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269911, upper bound: 315.1269882
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269911, upper bound: 315.1269882
time: 6.91 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269773, upper bound: 315.1270284
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269773, upper bound: 315.1270284
time: 6.94 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270284, upper bound: 315.1269773
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270284, upper bound: 315.1269773
time: 5.93 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269882, upper bound: 315.1269911
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269882, upper bound: 315.1269911
time: 5.58 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270284, upper bound: 315.1269773
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270284, upper bound: 315.1269773
time: 5.82 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269882, upper bound: 315.1269911
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269882, upper bound: 315.1269911
time: 5.18 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270031, upper bound: 315.1269867
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270031, upper bound: 315.1269867
time: 7.05 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269830, upper bound: 315.1270286
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269830, upper bound: 315.1270286
time: 6.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270031, upper bound: 315.1269867
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270031, upper bound: 315.1269867
time: 6.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269830, upper bound: 315.1270286
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1269830, upper bound: 315.1270286
time: 6.56 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270286, upper bound: 315.1269830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270286, upper bound: 315.1269830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269867, upper bound: 315.1270031
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269867, upper bound: 315.1270031
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270286, upper bound: 315.1269830
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270286, upper bound: 315.1269830
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269867, upper bound: 315.1270031
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269867, upper bound: 315.1270031
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269911, upper bound: 315.1269882
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269911, upper bound: 315.1269882
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269773, upper bound: 315.1270284
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269773, upper bound: 315.1270284
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269911, upper bound: 315.1269882
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269911, upper bound: 315.1269882
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269773, upper bound: 315.1270284
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269773, upper bound: 315.1270284
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270284, upper bound: 315.1269773
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270284, upper bound: 315.1269773
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269882, upper bound: 315.1269911
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269882, upper bound: 315.1269911
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270284, upper bound: 315.1269773
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270284, upper bound: 315.1269773
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269882, upper bound: 315.1269911
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269882, upper bound: 315.1269911
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270031, upper bound: 315.1269867
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270031, upper bound: 315.1269867
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269830, upper bound: 315.1270286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269830, upper bound: 315.1270286
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270031, upper bound: 315.1269867
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1270031, upper bound: 315.1269867
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269830, upper bound: 315.1270286
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.68
Output dim: 6, lower bound: -315.1269830, upper bound: 315.1270286
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=317.01361083984375
rel_dist={6: [-315.19662171336495, 315.19662171336495]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1965492, upper bound: 315.1966547
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1966547, upper bound: 315.1965492
time: 8.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.15
Output dim: 6, lower bound: -315.1965492, upper bound: 315.1966547
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.15
Output dim: 6, lower bound: -315.1966547, upper bound: 315.1965492

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1922952, upper bound: 315.1923153
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1922952, upper bound: 315.1923153
time: 5.94 seconds

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
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1923153, upper bound: 315.1922952
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1923153, upper bound: 315.1922952
time: 7.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.28
Output dim: 6, lower bound: -315.1922952, upper bound: 315.1923153
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.28
Output dim: 6, lower bound: -315.1922952, upper bound: 315.1923153
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.28
Output dim: 6, lower bound: -315.1923153, upper bound: 315.1922952
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.28
Output dim: 6, lower bound: -315.1923153, upper bound: 315.1922952

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901214, upper bound: 315.1901753
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901215, upper bound: 315.1901753
time: 7.91 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901214, upper bound: 315.1901753
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901214, upper bound: 315.1901754
time: 7.83 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901753, upper bound: 315.1901215
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901753, upper bound: 315.1901214
time: 8.20 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901754, upper bound: 315.1901214
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901753, upper bound: 315.1901215
time: 7.05 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.39
Output dim: 6, lower bound: -315.1901214, upper bound: 315.1901753
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.39
Output dim: 6, lower bound: -315.1901215, upper bound: 315.1901753
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.39
Output dim: 6, lower bound: -315.1901214, upper bound: 315.1901753
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.39
Output dim: 6, lower bound: -315.1901214, upper bound: 315.1901754
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.39
Output dim: 6, lower bound: -315.1901753, upper bound: 315.1901215
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.39
Output dim: 6, lower bound: -315.1901753, upper bound: 315.1901214
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.39
Output dim: 6, lower bound: -315.1901754, upper bound: 315.1901214
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.39
Output dim: 6, lower bound: -315.1901753, upper bound: 315.1901215

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809557, upper bound: 315.1809669
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809506, upper bound: 315.1809669
time: 6.77 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809557, upper bound: 315.1809669
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809506, upper bound: 315.1809669
time: 6.65 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809557, upper bound: 315.1809669
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809506, upper bound: 315.1809669
time: 7.50 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809557, upper bound: 315.1809669
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809506, upper bound: 315.1809669
time: 6.87 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809506
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809557
time: 7.67 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809506
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809557
time: 6.99 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809506
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809557
time: 7.04 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809506
time: 8.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809557
time: 7.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809557, upper bound: 315.1809669
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809506, upper bound: 315.1809669
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809557, upper bound: 315.1809669
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809506, upper bound: 315.1809669
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809557, upper bound: 315.1809669
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809506, upper bound: 315.1809669
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809557, upper bound: 315.1809669
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809506, upper bound: 315.1809669
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809506
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809557
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809506
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809557
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809506
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809557
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809506
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.73
Output dim: 6, lower bound: -315.1809669, upper bound: 315.1809557

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270734, upper bound: 315.1270178
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270734, upper bound: 315.1270178
time: 5.87 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270224, upper bound: 315.1270425
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270224, upper bound: 315.1270425
time: 6.57 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270734, upper bound: 315.1270178
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270734, upper bound: 315.1270178
time: 5.93 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270224, upper bound: 315.1270425
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270224, upper bound: 315.1270425
time: 6.75 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270281, upper bound: 315.1270238
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270281, upper bound: 315.1270238
time: 6.45 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270111, upper bound: 315.1270733
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270111, upper bound: 315.1270733
time: 5.16 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270281, upper bound: 315.1270238
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270281, upper bound: 315.1270238
time: 5.65 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270111, upper bound: 315.1270733
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270111, upper bound: 315.1270733
time: 5.33 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270733, upper bound: 315.1270111
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270733, upper bound: 315.1270111
time: 5.92 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270238, upper bound: 315.1270281
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270238, upper bound: 315.1270281
time: 5.75 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270733, upper bound: 315.1270111
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270733, upper bound: 315.1270111
time: 5.94 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270238, upper bound: 315.1270281
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270238, upper bound: 315.1270281
time: 5.26 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270425, upper bound: 315.1270224
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270425, upper bound: 315.1270224
time: 6.35 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270178, upper bound: 315.1270734
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270178, upper bound: 315.1270734
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270425, upper bound: 315.1270224
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270425, upper bound: 315.1270224
time: 6.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270178, upper bound: 315.1270734
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270178, upper bound: 315.1270734
time: 5.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270734, upper bound: 315.1270178
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270734, upper bound: 315.1270178
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270224, upper bound: 315.1270425
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270224, upper bound: 315.1270425
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270734, upper bound: 315.1270178
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270734, upper bound: 315.1270178
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270224, upper bound: 315.1270425
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270224, upper bound: 315.1270425
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270281, upper bound: 315.1270238
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270281, upper bound: 315.1270238
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270111, upper bound: 315.1270733
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270111, upper bound: 315.1270733
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270281, upper bound: 315.1270238
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270281, upper bound: 315.1270238
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270111, upper bound: 315.1270733
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270111, upper bound: 315.1270733
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270733, upper bound: 315.1270111
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270733, upper bound: 315.1270111
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270238, upper bound: 315.1270281
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270238, upper bound: 315.1270281
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270733, upper bound: 315.1270111
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270733, upper bound: 315.1270111
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270238, upper bound: 315.1270281
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270238, upper bound: 315.1270281
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270425, upper bound: 315.1270224
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270425, upper bound: 315.1270224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270178, upper bound: 315.1270734
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270178, upper bound: 315.1270734
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270425, upper bound: 315.1270224
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270425, upper bound: 315.1270224
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270178, upper bound: 315.1270734
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.26
Output dim: 6, lower bound: -315.1270178, upper bound: 315.1270734
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=317.01361083984375
rel_dist={6: [-315.19665474248563, 315.1966547424855]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1965566, upper bound: 315.1966706
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1966706, upper bound: 315.1965565
time: 6.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.87 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.87
Output dim: 6, lower bound: -315.1965566, upper bound: 315.1966706
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.87
Output dim: 6, lower bound: -315.1966706, upper bound: 315.1965565

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1923120, upper bound: 315.1923335
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1923120, upper bound: 315.1923335
time: 6.74 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1923335, upper bound: 315.1923120
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1923335, upper bound: 315.1923120
time: 6.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.24
Output dim: 6, lower bound: -315.1923120, upper bound: 315.1923335
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.24
Output dim: 6, lower bound: -315.1923120, upper bound: 315.1923335
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.24
Output dim: 6, lower bound: -315.1923335, upper bound: 315.1923120
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.24
Output dim: 6, lower bound: -315.1923335, upper bound: 315.1923120

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901436, upper bound: 315.1901999
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901436, upper bound: 315.1901999
time: 6.45 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901436, upper bound: 315.1901999
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901436, upper bound: 315.1901999
time: 6.42 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901999, upper bound: 315.1901436
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901999, upper bound: 315.1901436
time: 6.12 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901999, upper bound: 315.1901436
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1901999, upper bound: 315.1901436
time: 6.51 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -315.1901436, upper bound: 315.1901999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -315.1901436, upper bound: 315.1901999
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -315.1901436, upper bound: 315.1901999
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -315.1901436, upper bound: 315.1901999
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -315.1901999, upper bound: 315.1901436
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -315.1901999, upper bound: 315.1901436
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -315.1901999, upper bound: 315.1901436
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -315.1901999, upper bound: 315.1901436

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809674, upper bound: 315.1809796
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809620, upper bound: 315.1809797
time: 7.34 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809674, upper bound: 315.1809796
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809620, upper bound: 315.1809797
time: 6.79 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809674, upper bound: 315.1809796
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809620, upper bound: 315.1809797
time: 6.18 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809674, upper bound: 315.1809796
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809620, upper bound: 315.1809797
time: 6.43 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809797, upper bound: 315.1809620
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809796, upper bound: 315.1809674
time: 6.92 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809797, upper bound: 315.1809620
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809796, upper bound: 315.1809674
time: 6.45 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809797, upper bound: 315.1809620
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809796, upper bound: 315.1809674
time: 6.58 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809797, upper bound: 315.1809620
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809796, upper bound: 315.1809674
time: 6.52 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809674, upper bound: 315.1809796
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809620, upper bound: 315.1809797
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809674, upper bound: 315.1809796
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809620, upper bound: 315.1809797
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809674, upper bound: 315.1809796
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809620, upper bound: 315.1809797
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809674, upper bound: 315.1809796
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809620, upper bound: 315.1809797
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809797, upper bound: 315.1809620
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809796, upper bound: 315.1809674
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809797, upper bound: 315.1809620
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809796, upper bound: 315.1809674
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809797, upper bound: 315.1809620
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809796, upper bound: 315.1809674
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809797, upper bound: 315.1809620
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.50
Output dim: 6, lower bound: -315.1809796, upper bound: 315.1809674

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270955, upper bound: 315.1270352
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270955, upper bound: 315.1270352
time: 6.13 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270402, upper bound: 315.1270610
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270402, upper bound: 315.1270610
time: 5.51 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270955, upper bound: 315.1270352
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270955, upper bound: 315.1270352
time: 6.79 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270402, upper bound: 315.1270610
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270402, upper bound: 315.1270610
time: 6.31 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270466, upper bound: 315.1270416
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270466, upper bound: 315.1270416
time: 6.20 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270279, upper bound: 315.1270953
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270279, upper bound: 315.1270953
time: 5.49 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270466, upper bound: 315.1270416
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270466, upper bound: 315.1270416
time: 6.59 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270279, upper bound: 315.1270953
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270279, upper bound: 315.1270953
time: 5.76 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270953, upper bound: 315.1270279
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270953, upper bound: 315.1270279
time: 6.35 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270416, upper bound: 315.1270466
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270416, upper bound: 315.1270466
time: 6.20 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270953, upper bound: 315.1270279
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270953, upper bound: 315.1270279
time: 6.49 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270416, upper bound: 315.1270466
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270416, upper bound: 315.1270466
time: 5.49 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270610, upper bound: 315.1270402
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270610, upper bound: 315.1270402
time: 6.13 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270352, upper bound: 315.1270955
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270352, upper bound: 315.1270955
time: 6.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270610, upper bound: 315.1270402
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270610, upper bound: 315.1270402
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270352, upper bound: 315.1270955
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1270352, upper bound: 315.1270955
time: 6.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270955, upper bound: 315.1270352
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270955, upper bound: 315.1270352
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270402, upper bound: 315.1270610
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270402, upper bound: 315.1270610
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270955, upper bound: 315.1270352
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270955, upper bound: 315.1270352
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270402, upper bound: 315.1270610
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270402, upper bound: 315.1270610
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270466, upper bound: 315.1270416
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270466, upper bound: 315.1270416
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270279, upper bound: 315.1270953
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270279, upper bound: 315.1270953
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270466, upper bound: 315.1270416
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270466, upper bound: 315.1270416
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270279, upper bound: 315.1270953
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270279, upper bound: 315.1270953
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270953, upper bound: 315.1270279
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270953, upper bound: 315.1270279
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270416, upper bound: 315.1270466
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270416, upper bound: 315.1270466
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270953, upper bound: 315.1270279
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270953, upper bound: 315.1270279
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270416, upper bound: 315.1270466
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270416, upper bound: 315.1270466
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270610, upper bound: 315.1270402
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270610, upper bound: 315.1270402
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270352, upper bound: 315.1270955
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270352, upper bound: 315.1270955
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270610, upper bound: 315.1270402
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270610, upper bound: 315.1270402
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270352, upper bound: 315.1270955
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.18
Output dim: 6, lower bound: -315.1270352, upper bound: 315.1270955
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=317.01361083984375
rel_dist={6: [-315.19667063929785, 315.1966706328478]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 1862.01 seconds
