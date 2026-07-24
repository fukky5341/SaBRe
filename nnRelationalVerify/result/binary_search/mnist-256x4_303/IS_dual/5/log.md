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
execution time: IAR + LP analysis = 1.23 + 9.17 = 10.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -315.1966706, upper bound: 315.1966706


# Binary Search by BASE starts (time budget: 2689.60 seconds, max iter: 100)

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
Binary search time: 42.15 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2647.44 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1731408, upper bound: 315.1797323
time: 7.11 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1965687, upper bound: 315.1965687
time: 6.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.98 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.98
Output dim: 6, lower bound: -315.1731408, upper bound: 315.1797323
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.98
Output dim: 6, lower bound: -315.1965687, upper bound: 315.1965687

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -185.7228241, 148.8169098, -174.8029938, 140.1038818, -325.8266907, 323.6199036
1: -156.3873444, 131.2369843, -147.0469666, 123.4979477, -279.8852844, 278.2839355
2: -205.0139618, 133.8718109, -192.7902069, 125.9055634, -330.9195251, 326.6620178
3: -216.7644043, 114.7373581, -203.9159546, 107.8860626, -324.6504517, 318.6533203
4: -199.9647064, 152.8828888, -188.1284637, 143.7890625, -343.7537842, 341.0113525
5: -178.7982178, 139.5089569, -168.2542877, 131.2348175, -310.0329895, 307.7632446
6: -170.9838715, 164.6737061, -160.8820648, 154.8949432, -325.8788147, 325.5557251
7: -186.0695953, 156.6402435, -174.9291534, 147.3772736, -333.4468384, 331.5693970
8: -226.4187927, 156.1277313, -212.9881744, 146.8520966, -373.2707825, 369.1159058
9: -169.4191895, 167.4749146, -159.3720245, 157.5027466, -326.9219360, 326.8469238

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1655386, upper bound: 315.1711287
time: 8.02 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1651173, upper bound: 315.1709081
time: 7.97 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -175.0859222, 140.3284149, -175.4864807, 140.6464844, -315.7324219, 315.8148804
1: -147.2828369, 123.6969376, -147.6187286, 123.9792175, -271.2620544, 271.3156128
2: -193.1013489, 126.1062393, -193.5413055, 126.3884354, -319.4897766, 319.6475525
3: -204.2499084, 108.0594635, -204.7216034, 108.3048859, -312.5547791, 312.7810364
4: -188.4340057, 144.0214691, -188.8656311, 144.3499298, -332.7839355, 332.8870850
5: -168.5265350, 131.4449005, -168.9125214, 131.7426605, -300.2691956, 300.3573914
6: -161.1438446, 155.1462097, -161.5124664, 155.5011597, -316.6450195, 316.6585999
7: -175.2112885, 147.6148682, -175.6107178, 147.9506073, -323.1618958, 323.2255859
8: -213.3312836, 147.0843353, -213.8157196, 147.4116516, -360.7429199, 360.9000549
9: -159.6303558, 157.7554932, -159.9953308, 158.1130371, -317.7434082, 317.7508240

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1539696, upper bound: 315.1644475
time: 9.23 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1451145, upper bound: 315.1451145
time: 6.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.32 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 17.32
Output dim: 6, lower bound: -315.1655386, upper bound: 315.1711287
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 17.32
Output dim: 6, lower bound: -315.1651173, upper bound: 315.1709081
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 17.32
Output dim: 6, lower bound: -315.1539696, upper bound: 315.1644475
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 17.32
Output dim: 6, lower bound: -315.1451145, upper bound: 315.1451145
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=317.01361083984375
rel_dist={6: [-315.19656872135204, 315.19656872135204]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1748935, upper bound: 315.1838323
time: 6.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1966217, upper bound: 315.1966217
time: 6.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.63
Output dim: 6, lower bound: -315.1748935, upper bound: 315.1838323
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.63
Output dim: 6, lower bound: -315.1966217, upper bound: 315.1966217

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -185.7228241, 148.8169098, -175.2275696, 140.4409637, -326.1637878, 324.0444946
1: -156.3873444, 131.2369843, -147.4021759, 123.7969208, -280.1842346, 278.6391296
2: -205.0139618, 133.8718109, -193.2568207, 126.2055359, -331.2194824, 327.1285706
3: -216.7644043, 114.7373581, -204.4164734, 108.1462250, -324.9105835, 319.1537476
4: -199.9647064, 152.8828888, -188.5864105, 144.1375122, -344.1022034, 341.4692993
5: -178.7982178, 139.5089569, -168.6631927, 131.5503235, -310.3485107, 308.1721497
6: -170.9838715, 164.6737061, -161.2736816, 155.2715454, -326.2554321, 325.9473877
7: -186.0695953, 156.6402435, -175.3525391, 147.7334442, -333.8029785, 331.9927979
8: -226.4187927, 156.1277313, -213.5022736, 147.1996765, -373.6184082, 369.6299744
9: -169.4191895, 167.4749146, -159.7592316, 157.8818665, -327.3010559, 327.2341309

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1674449, upper bound: 315.1749096
time: 7.54 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1669196, upper bound: 315.1746216
time: 8.13 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -175.0859222, 140.3284149, -175.4864807, 140.6464844, -315.7324219, 315.8148804
1: -147.2828369, 123.6969376, -147.6187286, 123.9792175, -271.2620544, 271.3156128
2: -193.1013489, 126.1062393, -193.5413055, 126.3884354, -319.4897766, 319.6475525
3: -204.2499084, 108.0594635, -204.7216034, 108.3048859, -312.5547791, 312.7810364
4: -188.4340057, 144.0214691, -188.8656311, 144.3499298, -332.7839355, 332.8870850
5: -168.5265350, 131.4449005, -168.9125214, 131.7426605, -300.2691956, 300.3573914
6: -161.1438446, 155.1462097, -161.5124664, 155.5011597, -316.6450195, 316.6585999
7: -175.2112885, 147.6148682, -175.6107178, 147.9506073, -323.1618958, 323.2255859
8: -213.3312836, 147.0843353, -213.8157196, 147.4116516, -360.7429199, 360.9000549
9: -159.6303558, 157.7554932, -159.9953308, 158.1130371, -317.7434082, 317.7508240

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1577099, upper bound: 315.1711074
time: 7.89 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1451881, upper bound: 315.1451881
time: 4.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.13 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 14.13
Output dim: 6, lower bound: -315.1674449, upper bound: 315.1749096
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 14.13
Output dim: 6, lower bound: -315.1669196, upper bound: 315.1746216
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 14.13
Output dim: 6, lower bound: -315.1577099, upper bound: 315.1711074
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 14.13
Output dim: 6, lower bound: -315.1451881, upper bound: 315.1451881

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -170.3620453, 136.5677948, -175.2275696, 140.4409637, -310.8030090, 311.7953491
1: -143.5523529, 120.4753952, -147.4021759, 123.7969208, -267.3492737, 267.8775635
2: -188.1752777, 122.8871384, -193.2568207, 126.2055359, -314.3807678, 316.1439209
3: -198.9696960, 105.4101486, -204.4164734, 108.1462250, -307.1158752, 309.8265381
4: -183.6068878, 140.3180237, -188.5864105, 144.1375122, -327.7443848, 328.9044189
5: -164.0066833, 127.9285049, -168.6631927, 131.5503235, -295.5568848, 296.5916748
6: -157.0414886, 151.2255859, -161.2736816, 155.2715454, -312.3130493, 312.4992676
7: -170.7066650, 143.7254333, -175.3525391, 147.7334442, -318.4401245, 319.0779724
8: -207.9696350, 143.3427124, -213.5022736, 147.1996765, -355.1693115, 356.8449707
9: -155.4323883, 153.7477112, -159.7592316, 157.8818665, -313.3142090, 313.5068970

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1508288, upper bound: 315.1629177
time: 8.17 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1495317, upper bound: 315.1601988
time: 7.78 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -171.1837769, 137.2043457, -174.9981537, 140.2581635, -311.4419556, 312.2025146
1: -144.1493225, 120.9885788, -147.2094421, 123.6354141, -267.7847290, 268.1979675
2: -188.9866333, 123.3906479, -193.0043793, 126.0412445, -315.0278931, 316.3950195
3: -199.8446503, 105.8170471, -204.1488190, 108.0056000, -307.8502502, 309.9658813
4: -184.3568268, 140.8604736, -188.3403625, 143.9485626, -328.3053894, 329.2008057
5: -164.7817230, 128.4664154, -168.4421844, 131.3773651, -296.1590271, 296.9085999
6: -157.7734222, 151.8567810, -161.0640869, 155.0691223, -312.8425293, 312.9208679
7: -171.4184570, 144.3433685, -175.1221924, 147.5400238, -318.9584961, 319.4655762
8: -208.8715363, 143.9346313, -213.2256470, 147.0087433, -355.8801880, 357.1602478
9: -156.0802765, 154.3737335, -159.5493927, 157.6755981, -313.7558594, 313.9230957

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 52

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1485484, upper bound: 315.1614955
time: 7.87 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1473659, upper bound: 315.1589070
time: 7.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.97 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 18.97
Output dim: 6, lower bound: -315.1508288, upper bound: 315.1629177
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 18.97
Output dim: 6, lower bound: -315.1495317, upper bound: 315.1601988
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 18.97
Output dim: 6, lower bound: -315.1485484, upper bound: 315.1614955
IS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 18.97
Output dim: 6, lower bound: -315.1473659, upper bound: 315.1589070
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=317.01361083984375
rel_dist={6: [-315.19662171336495, 315.19662171336495]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1759257, upper bound: 315.1860037
time: 7.39 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1966547, upper bound: 315.1966547
time: 7.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.04 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.04
Output dim: 6, lower bound: -315.1759257, upper bound: 315.1860037
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.04
Output dim: 6, lower bound: -315.1966547, upper bound: 315.1966547

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -185.7228241, 148.8169098, -175.3933868, 140.5726166, -326.2954102, 324.2102966
1: -156.3873444, 131.2369843, -147.5408630, 123.9136734, -280.3010254, 278.7778320
2: -205.0139618, 133.8718109, -193.4390259, 126.3226852, -331.3366394, 327.3108521
3: -216.7644043, 114.7373581, -204.6118927, 108.2478561, -325.0122681, 319.3492126
4: -199.9647064, 152.8828888, -188.7652588, 144.2735901, -344.2382812, 341.6481323
5: -178.7982178, 139.5089569, -168.8229065, 131.6735077, -310.4717102, 308.3318481
6: -170.9838715, 164.6737061, -161.4266052, 155.4186249, -326.4024963, 326.1003113
7: -186.0695953, 156.6402435, -175.5179138, 147.8725586, -333.9421082, 332.1581421
8: -226.4187927, 156.1277313, -213.7030640, 147.3354645, -373.7541809, 369.8307495
9: -169.4191895, 167.4749146, -159.9104614, 158.0299225, -327.4490967, 327.3853149

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1685187, upper bound: 315.1770920
time: 7.84 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1679218, upper bound: 315.1766863
time: 6.90 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -175.0859222, 140.3284149, -175.4864807, 140.6464844, -315.7324219, 315.8148804
1: -147.2828369, 123.6969376, -147.6187286, 123.9792175, -271.2620544, 271.3156128
2: -193.1013489, 126.1062393, -193.5413055, 126.3884354, -319.4897766, 319.6475525
3: -204.2499084, 108.0594635, -204.7216034, 108.3048859, -312.5547791, 312.7810364
4: -188.4340057, 144.0214691, -188.8656311, 144.3499298, -332.7839355, 332.8870850
5: -168.5265350, 131.4449005, -168.9125214, 131.7426605, -300.2691956, 300.3573914
6: -161.1438446, 155.1462097, -161.5124664, 155.5011597, -316.6450195, 316.6585999
7: -175.2112885, 147.6148682, -175.6107178, 147.9506073, -323.1618958, 323.2255859
8: -213.3312836, 147.0843353, -213.8157196, 147.4116516, -360.7429199, 360.9000549
9: -159.6303558, 157.7554932, -159.9953308, 158.1130371, -317.7434082, 317.7508240

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1599670, upper bound: 315.1748383
time: 8.14 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1452308, upper bound: 315.1452308
time: 7.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.67 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 16.67
Output dim: 6, lower bound: -315.1685187, upper bound: 315.1770920
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 16.67
Output dim: 6, lower bound: -315.1679218, upper bound: 315.1766863
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.67
Output dim: 6, lower bound: -315.1599670, upper bound: 315.1748383
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 16.67
Output dim: 6, lower bound: -315.1452308, upper bound: 315.1452308

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -170.3620453, 136.5677948, -175.3933868, 140.5726166, -310.9346313, 311.9611816
1: -143.5523529, 120.4753952, -147.5408630, 123.9136734, -267.4660339, 268.0162659
2: -188.1752777, 122.8871384, -193.4390259, 126.3226852, -314.4978943, 316.3261719
3: -198.9696960, 105.4101486, -204.6118927, 108.2478561, -307.2175293, 310.0220032
4: -183.6068878, 140.3180237, -188.7652588, 144.2735901, -327.8804932, 329.0832825
5: -164.0066833, 127.9285049, -168.8229065, 131.6735077, -295.6801147, 296.7513733
6: -157.0414886, 151.2255859, -161.4266052, 155.4186249, -312.4601135, 312.6521912
7: -170.7066650, 143.7254333, -175.5179138, 147.8725586, -318.5792236, 319.2433167
8: -207.9696350, 143.3427124, -213.7030640, 147.3354645, -355.3050537, 357.0457458
9: -155.4323883, 153.7477112, -159.9104614, 158.0299225, -313.4622498, 313.6581116

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1530566, upper bound: 315.1672081
time: 7.76 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1516291, upper bound: 315.1639926
time: 6.33 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -171.1837769, 137.2043457, -175.3933868, 140.5726166, -311.7564087, 312.5977173
1: -144.1493225, 120.9885788, -147.5408630, 123.9136734, -268.0629883, 268.5293884
2: -188.9866333, 123.3906479, -193.4390259, 126.3226852, -315.3093262, 316.8296814
3: -199.8446503, 105.8170471, -204.6118927, 108.2478561, -308.0924988, 310.4289246
4: -184.3568268, 140.8604736, -188.7652588, 144.2735901, -328.6304321, 329.6257324
5: -164.7817230, 128.4664154, -168.8229065, 131.6735077, -296.4551697, 297.2893066
6: -157.7734222, 151.8567810, -161.4266052, 155.4186249, -313.1920166, 313.2833862
7: -171.4184570, 144.3433685, -175.5179138, 147.8725586, -319.2910156, 319.8612671
8: -208.8715363, 143.9346313, -213.7030640, 147.3354645, -356.2069702, 357.6376343
9: -156.0802765, 154.3737335, -159.9104614, 158.0299225, -314.1101685, 314.2841492

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1507075, upper bound: 315.1656298
time: 11.20 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1493521, upper bound: 315.1625458
time: 10.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -175.0859222, 140.3284149, -174.8934937, 140.1756592, -315.2615662, 315.2218628
1: -147.2828369, 123.6969376, -147.1196594, 123.5619507, -270.8447876, 270.8165894
2: -193.1013489, 126.1062393, -192.8945465, 125.9680405, -319.0693970, 319.0007629
3: -204.2499084, 108.0594635, -204.0393524, 107.9397964, -312.1896973, 312.0988159
4: -188.4340057, 144.0214691, -188.2272186, 143.8659821, -332.2999878, 332.2486572
5: -168.5265350, 131.4449005, -168.3426666, 131.3000488, -299.8265686, 299.7875671
6: -161.1438446, 155.1462097, -160.9718170, 154.9765472, -316.1203308, 316.1180420
7: -175.2112885, 147.6148682, -175.0229492, 147.4557800, -322.6670532, 322.6378174
8: -213.3312836, 147.0843353, -213.1022339, 146.9213257, -360.2526245, 360.1865845
9: -159.6303558, 157.7554932, -159.4595490, 157.5834503, -317.2138062, 317.2150269

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1587959, upper bound: 315.1716830
time: 11.06 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1589256, upper bound: 315.1722795
time: 11.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.45 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 23.45
Output dim: 6, lower bound: -315.1530566, upper bound: 315.1672081
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 23.45
Output dim: 6, lower bound: -315.1516291, upper bound: 315.1639926
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 23.45
Output dim: 6, lower bound: -315.1507075, upper bound: 315.1656298
IS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 23.45
Output dim: 6, lower bound: -315.1493521, upper bound: 315.1625458
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 23.45
Output dim: 6, lower bound: -315.1587959, upper bound: 315.1716830
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 23.45
Output dim: 6, lower bound: -315.1589256, upper bound: 315.1722795
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=317.01361083984375
rel_dist={6: [-315.19665474248563, 315.1966547424855]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1763818, upper bound: 315.1868779
time: 6.93 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1966706, upper bound: 315.1966706
time: 8.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.15
Output dim: 6, lower bound: -315.1763818, upper bound: 315.1868779
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.15
Output dim: 6, lower bound: -315.1966706, upper bound: 315.1966706

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -185.7228241, 148.8169098, -175.4572906, 140.6233368, -326.3461609, 324.2742004
1: -156.3873444, 131.2369843, -147.5943146, 123.9586716, -280.3460083, 278.8312378
2: -205.0139618, 133.8718109, -193.5092468, 126.3678207, -331.3817444, 327.3810425
3: -216.7644043, 114.7373581, -204.6872253, 108.2870026, -325.0513306, 319.4245300
4: -199.9647064, 152.8828888, -188.8341370, 144.3259888, -344.2907104, 341.7170410
5: -178.7982178, 139.5089569, -168.8844452, 131.7209778, -310.5191956, 308.3934021
6: -170.9838715, 164.6737061, -161.4855347, 155.4752655, -326.4591370, 326.1592407
7: -186.0695953, 156.6402435, -175.5816193, 147.9261322, -333.9957275, 332.2218628
8: -226.4187927, 156.1277313, -213.7804108, 147.3877716, -373.8065186, 369.9081421
9: -169.4191895, 167.4749146, -159.9687347, 158.0869598, -327.5061646, 327.4436340

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1690180, upper bound: 315.1780190
time: 7.93 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1683939, upper bound: 315.1775212
time: 9.56 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -175.0859222, 140.3284149, -175.4864807, 140.6464844, -315.7324219, 315.8148804
1: -147.2828369, 123.6969376, -147.6187286, 123.9792175, -271.2620544, 271.3156128
2: -193.1013489, 126.1062393, -193.5413055, 126.3884354, -319.4897766, 319.6475525
3: -204.2499084, 108.0594635, -204.7216034, 108.3048859, -312.5547791, 312.7810364
4: -188.4340057, 144.0214691, -188.8656311, 144.3499298, -332.7839355, 332.8870850
5: -168.5265350, 131.4449005, -168.9125214, 131.7426605, -300.2691956, 300.3573914
6: -161.1438446, 155.1462097, -161.5124664, 155.5011597, -316.6450195, 316.6585999
7: -175.2112885, 147.6148682, -175.6107178, 147.9506073, -323.1618958, 323.2255859
8: -213.3312836, 147.0843353, -213.8157196, 147.4116516, -360.7429199, 360.9000549
9: -159.6303558, 157.7554932, -159.9953308, 158.1130371, -317.7434082, 317.7508240

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1935861, upper bound: 315.1951625
time: 9.88 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1923335, upper bound: 315.1923335
time: 6.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.64 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 17.64
Output dim: 6, lower bound: -315.1690180, upper bound: 315.1780190
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 17.64
Output dim: 6, lower bound: -315.1683939, upper bound: 315.1775212
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.64
Output dim: 6, lower bound: -315.1935861, upper bound: 315.1951625
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.64
Output dim: 6, lower bound: -315.1923335, upper bound: 315.1923335

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -170.3620453, 136.5677948, -175.4572906, 140.6233368, -310.9853821, 312.0250854
1: -143.5523529, 120.4753952, -147.5943146, 123.9586716, -267.5110168, 268.0697021
2: -188.1752777, 122.8871384, -193.5092468, 126.3678207, -314.5429688, 316.3963928
3: -198.9696960, 105.4101486, -204.6872253, 108.2870026, -307.2566223, 310.0973511
4: -183.6068878, 140.3180237, -188.8341370, 144.3259888, -327.9328613, 329.1521606
5: -164.0066833, 127.9285049, -168.8844452, 131.7209778, -295.7276001, 296.8128967
6: -157.0414886, 151.2255859, -161.4855347, 155.4752655, -312.5167542, 312.7110901
7: -170.7066650, 143.7254333, -175.5816193, 147.9261322, -318.6328125, 319.3070068
8: -207.9696350, 143.3427124, -213.7804108, 147.3877716, -355.3573914, 357.1231079
9: -155.4323883, 153.7477112, -159.9687347, 158.0869598, -313.5193481, 313.7164001

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1540985, upper bound: 315.1690717
time: 7.19 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1526290, upper bound: 315.1656676
time: 6.33 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -171.1837769, 137.2043457, -175.4572906, 140.6233368, -311.8071289, 312.6616211
1: -144.1493225, 120.9885788, -147.5943146, 123.9586716, -268.1080017, 268.5827942
2: -188.9866333, 123.3906479, -193.5092468, 126.3678207, -315.3544617, 316.8998718
3: -199.8446503, 105.8170471, -204.6872253, 108.2870026, -308.1316223, 310.5042419
4: -184.3568268, 140.8604736, -188.8341370, 144.3259888, -328.6828003, 329.6946106
5: -164.7817230, 128.4664154, -168.8844452, 131.7209778, -296.5026855, 297.3508301
6: -157.7734222, 151.8567810, -161.4855347, 155.4752655, -313.2486572, 313.3423157
7: -171.4184570, 144.3433685, -175.5816193, 147.9261322, -319.3446045, 319.9249878
8: -208.8715363, 143.9346313, -213.7804108, 147.3877716, -356.2593079, 357.7150269
9: -156.0802765, 154.3737335, -159.9687347, 158.0869598, -314.1672363, 314.3424377

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 165

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1517021, upper bound: 315.1675258
time: 7.61 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1502849, upper bound: 315.1641808
time: 6.15 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -175.0859222, 140.3284149, -172.4270325, 138.2237091, -313.3096313, 312.7554321
1: -147.2828369, 123.6969376, -145.0606232, 121.8312073, -269.1140442, 268.7575684
2: -193.1013489, 126.1062393, -190.1800079, 124.2212524, -317.3226013, 316.2862244
3: -204.2499084, 108.0594635, -201.1290436, 106.4317474, -310.6816406, 309.1885071
4: -188.4340057, 144.0214691, -185.5790253, 141.8539124, -330.2879028, 329.6004333
5: -168.5265350, 131.4449005, -165.9658203, 129.4698944, -297.9963989, 297.4106750
6: -161.1438446, 155.1462097, -158.6859436, 152.7940674, -313.9378967, 313.8321533
7: -175.2112885, 147.6148682, -172.5597076, 145.3858337, -320.5971069, 320.1745300
8: -213.3312836, 147.0843353, -210.1367493, 144.9147491, -358.2460327, 357.2210693
9: -159.6303558, 157.7554932, -157.2180939, 155.3855743, -315.0158997, 314.9735718

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 52

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1733276, upper bound: 315.1597740
time: 6.61 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1451082, upper bound: 315.1451522
time: 5.06 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -175.0859222, 140.3284149, -185.2022552, 148.3476105, -323.4335327, 325.5306702
1: -147.2828369, 123.6969376, -155.8048859, 130.7952118, -278.0780640, 279.5018005
2: -193.1013489, 126.1062393, -204.2640839, 133.3563080, -326.4576416, 330.3703003
3: -204.2499084, 108.0594635, -215.9397583, 114.2732849, -318.5231934, 323.9991760
4: -188.4340057, 144.0214691, -199.2066193, 152.3074188, -340.7414246, 343.2280884
5: -168.5265350, 131.4449005, -178.2333832, 139.0480652, -307.5745850, 309.6782532
6: -161.1438446, 155.1462097, -170.3453369, 164.0433960, -325.1872559, 325.4915466
7: -175.2112885, 147.6148682, -185.4124451, 156.1269379, -331.3382263, 333.0272827
8: -213.3312836, 147.0843353, -225.5631561, 155.4632568, -368.7944946, 372.6474609
9: -159.6303558, 157.7554932, -168.9636993, 166.8108673, -326.4411926, 326.7191467

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1832467, upper bound: 315.1844787
time: 6.36 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1816008, upper bound: 315.1816008
time: 5.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.59 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 13.59
Output dim: 6, lower bound: -315.1540985, upper bound: 315.1690717
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 13.59
Output dim: 6, lower bound: -315.1526290, upper bound: 315.1656676
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 13.59
Output dim: 6, lower bound: -315.1517021, upper bound: 315.1675258
IS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 13.59
Output dim: 6, lower bound: -315.1502849, upper bound: 315.1641808
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 13.59
Output dim: 6, lower bound: -315.1733276, upper bound: 315.1597740
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 13.59
Output dim: 6, lower bound: -315.1451082, upper bound: 315.1451522
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 6, lower bound: -315.1832467, upper bound: 315.1844787
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 6, lower bound: -315.1816008, upper bound: 315.1816008

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -175.0859222, 140.3284149, -169.1979828, 135.5853119, -310.6712341, 309.5263977
1: -147.2828369, 123.6969376, -142.4270630, 119.5831833, -266.8660278, 266.1239929
2: -193.1013489, 126.1062393, -186.7100220, 121.9026337, -315.0039673, 312.8162231
3: -204.2499084, 108.0594635, -197.3966827, 104.5566330, -308.8065491, 305.4560852
4: -188.4340057, 144.0214691, -182.1526794, 139.2098236, -327.6437683, 326.1741333
5: -168.5265350, 131.4449005, -162.8272400, 126.9849930, -295.5114746, 294.2721252
6: -161.1438446, 155.1462097, -155.8112946, 150.0246735, -311.1685181, 310.9574890
7: -175.2112885, 147.6148682, -169.3954163, 142.6654053, -317.8766785, 317.0102234
8: -213.3312836, 147.0843353, -206.3228607, 142.1408539, -355.4721375, 353.4071350
9: -159.6303558, 157.7554932, -154.3827362, 152.5039978, -312.1343384, 312.1382446

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1640645, upper bound: 315.1502628
time: 8.82 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1118813, upper bound: 315.1229291
time: 5.25 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -175.0859222, 140.3284149, -171.1810760, 137.1560059, -312.2419434, 311.5094910
1: -147.2828369, 123.6969376, -144.0136261, 120.9173431, -268.2001953, 267.7105713
2: -193.1013489, 126.1062393, -188.8180237, 123.2594299, -316.3607788, 314.9241943
3: -204.2499084, 108.0594635, -199.6154022, 105.6750565, -309.9249573, 307.6748352
4: -188.4340057, 144.0214691, -184.1634064, 140.7209320, -329.1549377, 328.1848450
5: -168.5265350, 131.4449005, -164.7211609, 128.4099579, -296.9364929, 296.1660461
6: -161.1438446, 155.1462097, -157.6092987, 151.6886597, -312.8325195, 312.7554626
7: -175.2112885, 147.6148682, -171.2909088, 144.2754517, -319.4867554, 318.9057617
8: -213.3312836, 147.0843353, -208.6547394, 143.7184753, -357.0497437, 355.7390747
9: -159.6303558, 157.7554932, -156.1089020, 154.1819153, -313.8122559, 313.8643799

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1501431, upper bound: 315.1255053
time: 6.23 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.0998083, upper bound: 315.0998083
time: 5.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.03 seconds
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 13.03
Output dim: 6, lower bound: -315.1640645, upper bound: 315.1502628
IS_A2_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 13.03
Output dim: 6, lower bound: -315.1118813, upper bound: 315.1229291
IS_A2_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 13.03
Output dim: 6, lower bound: -315.1501431, upper bound: 315.1255053
IS_A2_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 13.03
Output dim: 6, lower bound: -315.0998083, upper bound: 315.0998083
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=317.01361083984375
rel_dist={6: [-315.19667063929785, 315.1966706328478]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 426.30 seconds
