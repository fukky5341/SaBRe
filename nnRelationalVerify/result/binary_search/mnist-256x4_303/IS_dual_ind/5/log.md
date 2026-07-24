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
execution time: IAR + LP analysis = 1.23 + 9.26 = 10.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -315.1966706, upper bound: 315.1966706


# Binary Search by BASE starts (time budget: 2689.51 seconds, max iter: 100)

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
Binary search time: 49.35 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2640.16 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1884149, upper bound: 315.1874502
time: 10.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1866293, upper bound: 315.1866293
time: 7.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.68
Output dim: 6, lower bound: -315.1884149, upper bound: 315.1874502
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.68
Output dim: 6, lower bound: -315.1866293, upper bound: 315.1866293

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -159.8907318, 128.2099457, -175.4864807, 140.6464844, -300.5372314, 303.6964111
1: -134.5814209, 113.0495148, -147.6187286, 123.9792175, -258.5606384, 260.6682129
2: -176.4401398, 115.2283859, -193.5413055, 126.3884354, -302.8285828, 308.7696838
3: -186.6460419, 98.8347626, -204.7216034, 108.3048859, -294.9508972, 303.5563354
4: -172.2470398, 131.5870361, -188.8656311, 144.3499298, -316.5969849, 320.4526672
5: -153.8958435, 119.9840393, -168.9125214, 131.7426605, -285.6384888, 288.8965454
6: -147.3485260, 141.8417816, -161.5124664, 155.5011597, -302.8496704, 303.3542480
7: -160.0037842, 134.8331451, -175.6107178, 147.9506073, -307.9544067, 310.4438477
8: -195.0727539, 134.4256897, -213.8157196, 147.4116516, -342.4843750, 348.2413940
9: -145.7881622, 144.1698914, -159.9953308, 158.1130371, -303.9011841, 304.1652222

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1866293, upper bound: 315.1866293
time: 10.44 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1866293, upper bound: 315.1866293
time: 9.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -161.3736572, 129.3796387, -172.0301666, 137.8929443, -299.2666016, 301.4097900
1: -135.7423553, 114.0286865, -144.7157440, 121.5464325, -257.2887268, 258.7443848
2: -177.9850922, 116.2186508, -189.7383118, 123.9138870, -301.8989868, 305.9569702
3: -188.2836304, 99.6429520, -200.6898651, 106.1860504, -294.4696655, 300.3328247
4: -173.7100220, 132.6823883, -185.1579132, 141.5035095, -315.2135010, 317.8402710
5: -155.3129730, 121.0364609, -165.5832062, 129.1377106, -284.4506531, 286.6196594
6: -148.6860504, 143.0573425, -158.3556213, 152.4520264, -301.1380310, 301.4129639
7: -161.3879852, 136.0165405, -172.1394501, 145.0374603, -306.4253845, 308.1559753
8: -196.7814026, 135.5856171, -209.6484528, 144.5352783, -341.3166809, 345.2340698
9: -147.0498047, 145.3934479, -156.8334198, 155.0065155, -302.0563354, 302.2268677

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1824486, upper bound: 315.1840531
time: 10.63 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1814723, upper bound: 315.1814723
time: 11.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.61
Output dim: 6, lower bound: -315.1866293, upper bound: 315.1866293
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.61
Output dim: 6, lower bound: -315.1866293, upper bound: 315.1866293
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.61
Output dim: 6, lower bound: -315.1824486, upper bound: 315.1840531
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.61
Output dim: 6, lower bound: -315.1814723, upper bound: 315.1814723

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -159.8907318, 128.2099457, -159.8907318, 128.2099457, -288.1006775, 288.1006775
1: -134.5814209, 113.0495148, -134.5814209, 113.0495148, -247.6309357, 247.6309357
2: -176.4401398, 115.2283859, -176.4401398, 115.2283859, -291.6685181, 291.6685181
3: -186.6460419, 98.8347626, -186.6460419, 98.8347626, -285.4808044, 285.4808044
4: -172.2470398, 131.5870361, -172.2470398, 131.5870361, -303.8340454, 303.8340454
5: -153.8958435, 119.9840393, -153.8958435, 119.9840393, -273.8798828, 273.8798828
6: -147.3485260, 141.8417816, -147.3485260, 141.8417816, -289.1903076, 289.1903076
7: -160.0037842, 134.8331451, -160.0037842, 134.8331451, -294.8369141, 294.8369141
8: -195.0727539, 134.4256897, -195.0727539, 134.4256897, -329.4984436, 329.4984436
9: -145.7881622, 144.1698914, -145.7881622, 144.1698914, -289.9580688, 289.9580688

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1855956, upper bound: 315.1833016
time: 7.92 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1833709, upper bound: 315.1824433
time: 7.50 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -159.8907318, 128.2099457, -161.3736572, 129.3796387, -289.2703857, 289.5835876
1: -134.5814209, 113.0495148, -135.7423553, 114.0286865, -248.6101074, 248.7918701
2: -176.4401398, 115.2283859, -177.9850922, 116.2186508, -292.6587830, 293.2134705
3: -186.6460419, 98.8347626, -188.2836304, 99.6429520, -286.2890015, 287.1183472
4: -172.2470398, 131.5870361, -173.7100220, 132.6823883, -304.9294434, 305.2969971
5: -153.8958435, 119.9840393, -155.3129730, 121.0364609, -274.9323120, 275.2969971
6: -147.3485260, 141.8417816, -148.6860504, 143.0573425, -290.4058838, 290.5278320
7: -160.0037842, 134.8331451, -161.3879852, 136.0165405, -296.0203247, 296.2211304
8: -195.0727539, 134.4256897, -196.7814026, 135.5856171, -330.6583862, 331.2070923
9: -145.7881622, 144.1698914, -147.0498047, 145.3934479, -291.1816101, 291.2196960

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1855956, upper bound: 315.1833016
time: 11.41 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1833709, upper bound: 315.1824433
time: 15.80 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -161.3296051, 129.3447418, -168.9791565, 135.4767151, -296.8063354, 298.3239136
1: -135.7055511, 113.9977493, -142.1643829, 119.4042664, -255.1098022, 256.1621399
2: -177.9367065, 116.1874695, -186.3861084, 121.7526703, -299.6893311, 302.5735779
3: -188.2319031, 99.6159821, -197.1069946, 104.3178711, -292.5497742, 296.7229614
4: -173.6626892, 132.6464539, -181.8804779, 139.0145721, -312.6772156, 314.5268555
5: -155.2705383, 121.0037384, -162.6443634, 126.8710785, -282.1415405, 283.6480713
6: -148.6453552, 143.0183716, -155.5368500, 149.7521667, -298.3974915, 298.5551758
7: -161.3440552, 135.9796143, -169.0967712, 142.4795990, -303.8235779, 305.0763855
8: -196.7284393, 135.5496826, -205.9795532, 142.0455322, -338.7739868, 341.5292053
9: -147.0098114, 145.3542023, -154.0638580, 152.2864075, -299.2962036, 299.4179993

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1814723, upper bound: 315.1814723
time: 9.99 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1814723, upper bound: 315.1814723
time: 12.22 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -161.1406250, 129.1948700, -181.8156891, 145.6496429, -306.7902832, 311.0105591
1: -135.5474091, 113.8651276, -152.9615326, 128.4112244, -263.9585876, 266.8266296
2: -177.7287750, 116.0534821, -200.5386047, 130.9318390, -308.6606140, 316.5921021
3: -188.0095978, 99.4999237, -211.9893494, 112.1982422, -300.2078247, 311.4892578
4: -173.4595490, 132.4921265, -195.5742188, 149.5188599, -322.9783936, 328.0663452
5: -155.0885162, 120.8631363, -174.9704437, 136.4953461, -291.5838623, 295.8335876
6: -148.4709930, 142.8511658, -167.2515259, 161.0568695, -309.5278015, 310.1026917
7: -161.1552124, 135.8211212, -182.0123444, 153.2718353, -314.4270325, 317.8334656
8: -196.5010681, 135.3949127, -221.4832611, 152.6465302, -349.1475830, 356.8781128
9: -146.8380737, 145.1857605, -165.8668365, 163.7677612, -310.6058350, 311.0526123

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
time: 9.78 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1785834, upper bound: 315.1785834
time: 8.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.81 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.81
Output dim: 6, lower bound: -315.1855956, upper bound: 315.1833016
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.81
Output dim: 6, lower bound: -315.1833709, upper bound: 315.1824433
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.81
Output dim: 6, lower bound: -315.1855956, upper bound: 315.1833016
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.81
Output dim: 6, lower bound: -315.1833709, upper bound: 315.1824433
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.81
Output dim: 6, lower bound: -315.1814723, upper bound: 315.1814723
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.81
Output dim: 6, lower bound: -315.1814723, upper bound: 315.1814723
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.81
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.81
Output dim: 6, lower bound: -315.1785834, upper bound: 315.1785834

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -156.8226471, 125.7797165, -159.8465271, 128.1749115, -284.9974670, 285.6262512
1: -132.0159149, 110.8953323, -134.5445099, 113.0184555, -245.0343628, 245.4398499
2: -173.0692749, 113.0549240, -176.3915710, 115.1970673, -288.2662964, 289.4465027
3: -183.0433502, 96.9565201, -186.5941315, 98.8076935, -281.8509827, 283.5506592
4: -168.9519958, 129.0843811, -172.1995087, 131.5509491, -300.5029297, 301.2838440
5: -150.9402466, 117.7046280, -153.8532715, 119.9512024, -270.8913574, 271.5578918
6: -144.5147705, 139.1270294, -147.3076935, 141.8026581, -286.3174438, 286.4346619
7: -156.9443665, 132.2603149, -159.9596863, 134.7960815, -291.7404480, 292.2200012
8: -191.3835144, 131.9221802, -195.0195923, 134.3896179, -325.7731018, 326.9417725
9: -143.0031128, 141.4350433, -145.7480469, 144.1305237, -287.1336365, 287.1831055

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1854567, upper bound: 315.1828680
time: 11.78 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1859709, upper bound: 315.1842940
time: 10.32 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -169.1654358, 135.5594025, -159.6531219, 128.0214996, -297.1868896, 295.2124634
1: -142.4008331, 119.5611191, -134.3827057, 112.8828049, -255.2836304, 253.9438171
2: -186.6751099, 121.8805695, -176.1788177, 115.0599136, -301.7349854, 298.0593567
3: -197.3607788, 104.5366516, -186.3667297, 98.6889877, -296.0497742, 290.9033813
4: -182.1186676, 139.1839294, -171.9916077, 131.3930969, -313.5117798, 311.1755371
5: -162.7949524, 126.9594345, -153.6668854, 119.8072205, -282.6021729, 280.6262512
6: -155.7826996, 149.9965515, -147.1292877, 141.6316376, -297.4143372, 297.1258545
7: -169.3643341, 142.6382446, -159.7664032, 134.6338043, -303.9981384, 302.4046631
8: -206.2855377, 142.1148834, -194.7869263, 134.2312012, -340.5166931, 336.9017944
9: -154.3539581, 152.4757080, -145.5723114, 143.9581451, -298.3121033, 298.0479736

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1829780, upper bound: 315.1821880
time: 9.21 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1835861, upper bound: 315.1835861
time: 8.75 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -156.8226471, 125.7797165, -161.3296051, 129.3447418, -286.1672974, 287.1093140
1: -132.0159149, 110.8953323, -135.7055511, 113.9977493, -246.0136719, 246.6008759
2: -173.0692749, 113.0549240, -177.9367065, 116.1874695, -289.2567139, 290.9916382
3: -183.0433502, 96.9565201, -188.2319031, 99.6159821, -282.6593323, 285.1884155
4: -168.9519958, 129.0843811, -173.6626892, 132.6464539, -301.5983582, 302.7469788
5: -150.9402466, 117.7046280, -155.2705383, 121.0037384, -271.9439392, 272.9750977
6: -144.5147705, 139.1270294, -148.6453552, 143.0183716, -287.5331421, 287.7723694
7: -156.9443665, 132.2603149, -161.3440552, 135.9796143, -292.9239807, 293.6043701
8: -191.3835144, 131.9221802, -196.7284393, 135.5496826, -326.9331360, 328.6506348
9: -143.0031128, 141.4350433, -147.0098114, 145.3542023, -288.3572998, 288.4448547

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1833709, upper bound: 315.1824433
time: 9.28 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1833709, upper bound: 315.1824433
time: 7.39 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -169.1654358, 135.5594025, -161.1406250, 129.1948700, -298.3601990, 296.6999817
1: -142.4008331, 119.5611191, -135.5474091, 113.8651276, -256.2659607, 255.1085205
2: -186.6751099, 121.8805695, -177.7287750, 116.0534821, -302.7285767, 299.6093445
3: -197.3607788, 104.5366516, -188.0095978, 99.4999237, -296.8607178, 292.5462646
4: -182.1186676, 139.1839294, -173.4595490, 132.4921265, -314.6107788, 312.6434937
5: -162.7949524, 126.9594345, -155.0885162, 120.8631363, -283.6580811, 282.0478821
6: -155.7826996, 149.9965515, -148.4709930, 142.8511658, -298.6338196, 298.4675293
7: -169.3643341, 142.6382446, -161.1552124, 135.8211212, -305.1854553, 303.7934570
8: -206.2855377, 142.1148834, -196.5010681, 135.3949127, -341.6803284, 338.6159363
9: -154.3539581, 152.4757080, -146.8380737, 145.1857605, -299.5397339, 299.3137512

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1798449, upper bound: 315.1782833
time: 11.10 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1804604, upper bound: 315.1795233
time: 14.05 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -158.3348236, 126.9729843, -168.9791565, 135.4767151, -293.8115234, 295.9520874
1: -133.2000427, 111.8948898, -142.1643829, 119.4042664, -252.6043091, 254.0592651
2: -174.6463776, 114.0672150, -186.3861084, 121.7526703, -296.3990173, 300.4533081
3: -184.7152405, 97.7816238, -197.1069946, 104.3178711, -289.0331116, 294.8885803
4: -170.4481506, 130.2042694, -181.8804779, 139.0145721, -309.4627075, 312.0847168
5: -152.3862457, 118.7797012, -162.6443634, 126.8710785, -279.2573242, 281.4240417
6: -145.8793793, 140.3677521, -155.5368500, 149.7521667, -295.6314697, 295.9046021
7: -158.3582611, 133.4686279, -169.0967712, 142.4795990, -300.8377991, 302.5653992
8: -193.1271973, 133.1082916, -205.9795532, 142.0455322, -335.1727295, 339.0878296
9: -144.2917328, 142.6838379, -154.0638580, 152.2864075, -296.5781250, 296.7476807

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1788913, upper bound: 315.1798912
time: 11.52 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1796346, upper bound: 315.1814284
time: 9.84 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -171.0965881, 137.0865479, -168.9791565, 135.4767151, -306.5733032, 306.0657043
1: -143.9404907, 120.8581924, -142.1643829, 119.4042664, -263.3447266, 263.0225830
2: -188.7236176, 123.2000656, -186.3861084, 121.7526703, -310.4762878, 309.5861816
3: -199.5131989, 105.6205826, -197.1069946, 104.3178711, -303.8310547, 302.7275391
4: -184.0714874, 140.6521759, -181.8804779, 139.0145721, -323.0860596, 322.5326538
5: -164.6377716, 128.3446503, -162.6443634, 126.8710785, -291.5088196, 290.9889832
6: -157.5299988, 151.6110992, -155.5368500, 149.7521667, -307.2821655, 307.1479492
7: -171.2072449, 144.2008514, -169.0967712, 142.4795990, -313.6867981, 313.2975769
8: -208.5482025, 143.6468201, -205.9795532, 142.0455322, -350.5937500, 349.6263733
9: -156.0316620, 154.0987396, -154.0638580, 152.2864075, -308.3180542, 308.1625061

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1788913, upper bound: 315.1798912
time: 9.95 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1796346, upper bound: 315.1814284
time: 9.88 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -151.8849182, 121.8413544, -180.6007690, 144.6851044, -296.5699768, 302.4421387
1: -127.8988647, 107.4704132, -151.9573975, 127.5715179, -255.4703674, 259.4277649
2: -167.6065369, 109.4788895, -199.2104950, 130.0700989, -297.6766357, 308.6893921
3: -177.2960968, 93.9522781, -210.5832825, 111.4703217, -288.7663574, 304.5355225
4: -163.6575317, 124.9431000, -194.2876740, 148.5285492, -312.1860657, 319.2307739
5: -146.2171478, 113.9130402, -173.8063354, 135.5831146, -281.8002625, 287.7193604
6: -140.1015472, 134.8249817, -166.1530304, 160.0033264, -300.1048584, 300.9779968
7: -151.9131775, 128.0626221, -180.8000336, 152.2543640, -304.1675415, 308.8625793
8: -185.4225922, 127.8424225, -220.0298157, 151.6564178, -337.0790100, 347.8722229
9: -138.4196777, 137.0056458, -164.7625275, 162.6948853, -301.1145630, 301.7681580

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
time: 7.55 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
time: 7.02 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -155.2188721, 124.4811859, -181.8156891, 145.6496429, -300.8685303, 306.2968750
1: -130.6434479, 109.7616348, -152.9615326, 128.4112244, -259.0546265, 262.7231140
2: -171.2512054, 111.8447571, -200.5386047, 130.9318390, -302.1830139, 312.3833618
3: -181.1500702, 95.9411545, -211.9893494, 112.1982422, -293.3483276, 307.9305115
4: -167.1795349, 127.6565628, -195.5742188, 149.5188599, -316.6983337, 323.2307739
5: -149.4000397, 116.4044952, -174.9704437, 136.4953461, -285.8953857, 291.3749390
6: -143.1104889, 137.7032623, -167.2515259, 161.0568695, -304.1673279, 304.9547729
7: -155.2337036, 130.8485718, -182.0123444, 153.2718353, -308.5055542, 312.8609009
8: -189.4050903, 130.5517273, -221.4832611, 152.6465302, -342.0516357, 352.0349731
9: -141.4439697, 139.9403076, -165.8668365, 163.7677612, -305.2117004, 305.8071289

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1779742, upper bound: 315.1773191
time: 9.57 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1779742, upper bound: 315.1785834
time: 10.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.21 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1854567, upper bound: 315.1828680
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1859709, upper bound: 315.1842940
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1829780, upper bound: 315.1821880
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1835861, upper bound: 315.1835861
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1833709, upper bound: 315.1824433
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1833709, upper bound: 315.1824433
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1798449, upper bound: 315.1782833
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1804604, upper bound: 315.1795233
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1788913, upper bound: 315.1798912
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1796346, upper bound: 315.1814284
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1788913, upper bound: 315.1798912
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1796346, upper bound: 315.1814284
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1779742, upper bound: 315.1773191
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.21
Output dim: 6, lower bound: -315.1779742, upper bound: 315.1785834

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -155.6107178, 124.8170853, -150.3374176, 120.6243515, -276.2350159, 275.1545105
1: -131.0134735, 110.0570526, -126.6867142, 106.4496918, -237.4631653, 236.7437592
2: -171.7431793, 112.1943512, -165.9917755, 108.4471893, -280.1903687, 278.1860962
3: -181.6399994, 96.2298050, -175.5918427, 93.1063766, -274.7463684, 271.8216248
4: -167.6682587, 128.0960083, -162.1316223, 123.7991028, -291.4673462, 290.2276306
5: -149.7787323, 116.7942200, -144.7413177, 112.8126755, -262.5914001, 261.5355225
6: -143.4181061, 138.0754700, -138.7084351, 133.5533752, -276.9714966, 276.7838440
7: -155.7343445, 131.2444916, -150.4710236, 126.8272095, -282.5615234, 281.7155151
8: -189.9325409, 130.9337463, -183.6389771, 126.6339340, -316.5664673, 314.5726318
9: -141.9008484, 140.3636475, -137.1010742, 135.7254486, -277.6262817, 277.4647217

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1844561, upper bound: 315.1827275
time: 8.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1844561, upper bound: 315.1828680
time: 12.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -156.8226471, 125.7797165, -154.0558929, 123.5634537, -280.3859558, 279.8356018
1: -132.0159149, 110.8953323, -129.7473755, 109.0056763, -241.0215912, 240.6427002
2: -173.0692749, 113.0549240, -170.0573273, 111.0816803, -284.1509399, 283.1122437
3: -183.0433502, 96.9565201, -179.8859863, 95.3256149, -278.3689270, 276.8424988
4: -168.9519958, 129.0843811, -166.0541229, 126.8200150, -295.7720032, 295.1384277
5: -150.9402466, 117.7046280, -148.2904510, 115.5930710, -266.5332642, 265.9950867
6: -144.5147705, 139.1270294, -142.0647125, 136.7663879, -281.2811584, 281.1916809
7: -156.9443665, 132.2603149, -154.1669617, 129.9326477, -286.8770142, 286.4272766
8: -191.3835144, 131.9221802, -188.0792389, 129.6535950, -321.0370789, 320.0014038
9: -143.0031128, 141.4350433, -140.4710693, 138.9998474, -282.0029602, 281.9061279

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1845656, upper bound: 315.1836369
time: 8.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1845656, upper bound: 315.1842940
time: 7.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -167.9481354, 134.5928345, -150.1403809, 120.4680328, -288.4161377, 284.7331543
1: -141.3944855, 118.7195435, -126.5219803, 106.3114853, -247.7059631, 245.2415009
2: -185.3441620, 121.0169678, -165.7750244, 108.3074799, -293.6516418, 286.7919922
3: -195.9514923, 103.8074417, -175.3601990, 92.9854889, -288.9369812, 279.1675720
4: -180.8296204, 138.1917267, -161.9197998, 123.6382828, -304.4678955, 300.1115112
5: -161.6282349, 126.0451202, -144.5513763, 112.6660309, -274.2942505, 270.5964966
6: -154.6819611, 148.9409332, -138.5266876, 133.3791351, -288.0610962, 287.4676208
7: -168.1496582, 141.6184387, -150.2740784, 126.6619186, -294.8115845, 291.8925171
8: -204.8292084, 141.1224976, -183.4019012, 126.4725494, -331.3017578, 324.5244141
9: -153.2474823, 151.4002991, -136.9220734, 135.5498962, -288.7973633, 288.3223267

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1820620, upper bound: 315.1820620
time: 7.50 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1820620, upper bound: 315.1821880
time: 10.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -169.1654358, 135.5594025, -153.8612671, 123.4090805, -292.5744019, 289.4206543
1: -142.4008331, 119.5611191, -129.5846405, 108.8691864, -251.2700195, 249.1457520
2: -186.6751099, 121.8805695, -169.8432465, 110.9437027, -297.6188049, 291.7237549
3: -197.3607788, 104.5366516, -179.6571808, 95.2061768, -292.5669556, 284.1938171
4: -182.1186676, 139.1839294, -165.8449097, 126.6611786, -308.7798462, 305.0288391
5: -162.7949524, 126.9594345, -148.1029205, 115.4482346, -278.2431946, 275.0623474
6: -155.7826996, 149.9965515, -141.8851318, 136.5942383, -292.3768921, 291.8816833
7: -169.3643341, 142.6382446, -153.9724579, 129.7693939, -299.1337280, 296.6107178
8: -206.2855377, 142.1148834, -187.8451385, 129.4941559, -335.7796326, 329.9599915
9: -154.3539581, 152.4757080, -140.2942047, 138.8264465, -293.1804199, 292.7698669

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1821880, upper bound: 315.1829780
time: 9.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1821880, upper bound: 315.1835861
time: 10.07 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -156.8226471, 125.7797165, -158.3348236, 126.9729843, -283.7955017, 284.1145325
1: -132.0159149, 110.8953323, -133.2000427, 111.8948898, -243.9107971, 244.0953674
2: -173.0692749, 113.0549240, -174.6463776, 114.0672150, -287.1364441, 287.7012939
3: -183.0433502, 96.9565201, -184.7152405, 97.7816238, -280.8249512, 281.6717529
4: -168.9519958, 129.0843811, -170.4481506, 130.2042694, -299.1562500, 299.5325012
5: -150.9402466, 117.7046280, -152.3862457, 118.7797012, -269.7198792, 270.0908813
6: -144.5147705, 139.1270294, -145.8793793, 140.3677521, -284.8825073, 285.0063477
7: -156.9443665, 132.2603149, -158.3582611, 133.4686279, -290.4129944, 290.6185913
8: -191.3835144, 131.9221802, -193.1271973, 133.1082916, -324.4917908, 325.0493774
9: -143.0031128, 141.4350433, -144.2917328, 142.6838379, -285.6869507, 285.7267761

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1815256, upper bound: 315.1797149
time: 11.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1829920, upper bound: 315.1803887
time: 9.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -156.8226471, 125.7797165, -171.0965881, 137.0865479, -293.9091187, 296.8762817
1: -132.0159149, 110.8953323, -143.9404907, 120.8581924, -252.8741150, 254.8358154
2: -173.0692749, 113.0549240, -188.7236176, 123.2000656, -296.2693176, 301.7785339
3: -183.0433502, 96.9565201, -199.5131989, 105.6205826, -288.6638794, 296.4697266
4: -168.9519958, 129.0843811, -184.0714874, 140.6521759, -309.6041870, 313.1558533
5: -150.9402466, 117.7046280, -164.6377716, 128.3446503, -279.2848511, 282.3423767
6: -144.5147705, 139.1270294, -157.5299988, 151.6110992, -296.1258545, 296.6570435
7: -156.9443665, 132.2603149, -171.2072449, 144.2008514, -301.1452026, 303.4675598
8: -191.3835144, 131.9221802, -208.5482025, 143.6468201, -335.0303345, 340.4703979
9: -143.0031128, 141.4350433, -156.0316620, 154.0987396, -297.1018677, 297.4667053

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1815256, upper bound: 315.1797149
time: 8.26 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1829920, upper bound: 315.1803887
time: 8.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -167.9481354, 134.5928345, -151.8849182, 121.8413544, -289.7894897, 286.4777222
1: -141.3944855, 118.7195435, -127.8988647, 107.4704132, -248.8648682, 246.6183777
2: -185.3441620, 121.0169678, -167.6065369, 109.4788895, -294.8230286, 288.6235046
3: -195.9514923, 103.8074417, -177.2960968, 93.9522781, -289.9037781, 281.1035156
4: -180.8296204, 138.1917267, -163.6575317, 124.9431000, -305.7727051, 301.8492432
5: -161.6282349, 126.0451202, -146.2171478, 113.9130402, -275.5412598, 272.2622375
6: -154.6819611, 148.9409332, -140.1015472, 134.8249817, -289.5069580, 289.0424194
7: -168.1496582, 141.6184387, -151.9131775, 128.0626221, -296.2122803, 293.5316162
8: -204.8292084, 141.1224976, -185.4225922, 127.8424225, -332.6716309, 326.5451050
9: -153.2474823, 151.4002991, -138.4196777, 137.0056458, -290.2530823, 289.8199158

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1791146, upper bound: 315.1781991
time: 7.24 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1791146, upper bound: 315.1782833
time: 7.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -169.1654358, 135.5594025, -155.2188721, 124.4811859, -293.6465759, 290.7782593
1: -142.4008331, 119.5611191, -130.6434479, 109.7616348, -252.1624756, 250.2045593
2: -186.6751099, 121.8805695, -171.2512054, 111.8447571, -298.5198669, 293.1317444
3: -197.3607788, 104.5366516, -181.1500702, 95.9411545, -293.3019409, 285.6867065
4: -182.1186676, 139.1839294, -167.1795349, 127.6565628, -309.7752380, 306.3634644
5: -162.7949524, 126.9594345, -149.4000397, 116.4044952, -279.1994629, 276.3594360
6: -155.7826996, 149.9965515, -143.1104889, 137.7032623, -293.4859009, 293.1070557
7: -169.3643341, 142.6382446, -155.2337036, 130.8485718, -300.2128906, 297.8719482
8: -206.2855377, 142.1148834, -189.4050903, 130.5517273, -336.8371887, 331.5199585
9: -154.3539581, 152.4757080, -141.4439697, 139.9403076, -294.2942505, 293.9196167

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1792239, upper bound: 315.1790194
time: 7.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1792239, upper bound: 315.1795233
time: 7.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -157.1419220, 126.0249329, -159.4899750, 127.9391022, -285.0810242, 285.5148621
1: -132.2128601, 111.0694580, -134.3252411, 112.8485031, -245.0613708, 245.3946838
2: -173.3410034, 113.2198639, -176.0114136, 115.0178986, -288.3588257, 289.2312317
3: -183.3333893, 97.0661621, -186.1270752, 98.6318054, -281.9652100, 283.1932373
4: -169.1840210, 129.2309265, -171.8321533, 131.2766724, -300.4606934, 301.0630188
5: -151.2424774, 117.8837433, -153.5491638, 119.7464371, -270.9888306, 271.4328613
6: -144.7996521, 139.3326263, -146.9566803, 141.5245361, -286.3241577, 286.2893066
7: -157.1667175, 132.4684906, -159.6285858, 134.5286865, -291.6953735, 292.0970764
8: -191.6987915, 132.1347961, -194.6253510, 134.3049774, -326.0037842, 326.7601013
9: -143.2066650, 141.6291046, -145.4358826, 143.9026489, -287.1093140, 287.0650024

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1826092, upper bound: 315.1826092
time: 6.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1826092, upper bound: 315.1827907
time: 7.02 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -158.3348236, 126.9729843, -163.0884552, 130.7874451, -289.1222534, 290.0614319
1: -133.2000427, 111.8948898, -137.2873077, 115.3237610, -248.5238037, 249.1821899
2: -174.6463776, 114.0672150, -179.9449615, 117.5675735, -292.2139587, 294.0121460
3: -184.7152405, 97.7816238, -190.2853394, 100.7772827, -285.4924927, 288.0669556
4: -170.4481506, 130.2042694, -175.6303711, 134.2038422, -304.6519775, 305.8346252
5: -152.3862457, 118.7797012, -156.9884033, 122.4385910, -274.8248291, 275.7680969
6: -145.8793793, 140.3677521, -150.2050781, 144.6314545, -290.5108337, 290.5728149
7: -158.3582611, 133.4686279, -163.2070770, 137.5348511, -295.8930664, 296.6757202
8: -193.1271973, 133.1082916, -198.9222107, 137.2291718, -330.3563843, 332.0305176
9: -144.2917328, 142.6838379, -148.6976624, 147.0691681, -291.3609009, 291.3815002

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1827907, upper bound: 315.1836072
time: 6.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1827907, upper bound: 315.1844264
time: 7.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -169.8978119, 136.1340179, -159.4899750, 127.9391022, -297.8369141, 295.6239319
1: -142.9483490, 120.0290680, -134.3252411, 112.8485031, -255.7968445, 254.3543091
2: -187.4123077, 122.3490601, -176.0114136, 115.0178986, -302.4302063, 298.3604736
3: -198.1246643, 104.9017715, -186.1270752, 98.6318054, -296.7564697, 291.0288391
4: -182.8010101, 139.6741333, -171.8321533, 131.2766724, -314.0776978, 311.5062256
5: -163.4881439, 127.4442520, -153.5491638, 119.7464371, -283.2345581, 280.9934082
6: -156.4450989, 150.5706329, -146.9566803, 141.5245361, -297.9696350, 297.5273132
7: -170.0098419, 143.1959229, -159.6285858, 134.5286865, -304.5384827, 302.8245239
8: -207.1127472, 142.6690674, -194.6253510, 134.3049774, -341.4176636, 337.2943726
9: -154.9413757, 153.0391998, -145.4358826, 143.9026489, -298.8440247, 298.4750671

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1781545, upper bound: 315.1797806
time: 10.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1781545, upper bound: 315.1798912
time: 10.27 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -171.0965881, 137.0865479, -163.0884552, 130.7874451, -301.8840027, 300.1749878
1: -143.9404907, 120.8581924, -137.2873077, 115.3237610, -259.2642517, 258.1455078
2: -188.7236176, 123.2000656, -179.9449615, 117.5675735, -306.2911987, 303.1450195
3: -199.5131989, 105.6205826, -190.2853394, 100.7772827, -300.2904663, 295.9059143
4: -184.0714874, 140.6521759, -175.6303711, 134.2038422, -318.2753296, 316.2825317
5: -164.6377716, 128.3446503, -156.9884033, 122.4385910, -287.0763245, 285.3330688
6: -157.5299988, 151.6110992, -150.2050781, 144.6314545, -302.1614380, 301.8161621
7: -171.2072449, 144.2008514, -163.2070770, 137.5348511, -308.7420654, 307.4079285
8: -208.5482025, 143.6468201, -198.9222107, 137.2291718, -345.7773743, 342.5690308
9: -156.0316620, 154.0987396, -148.6976624, 147.0691681, -303.1007996, 302.7963867

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1783287, upper bound: 315.1808458
time: 11.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1783287, upper bound: 315.1814284
time: 7.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -151.8849182, 121.8413544, -167.9781799, 134.6167603, -286.5016174, 289.8194885
1: -127.8988647, 107.4704132, -141.4187469, 118.7398987, -246.6387482, 248.8891296
2: -167.6065369, 109.4788895, -185.3764343, 121.0373611, -288.6438904, 294.8553162
3: -177.2960968, 93.9522781, -195.9846497, 103.8258820, -281.1219788, 289.9369202
4: -163.6575317, 124.9431000, -180.8610535, 138.2156525, -301.8731689, 305.8041382
5: -146.2171478, 113.9130402, -161.6580048, 126.0686951, -272.2858276, 275.5710449
6: -140.1015472, 134.8249817, -154.7083435, 148.9669189, -289.0684509, 289.5333252
7: -151.9131775, 128.0626221, -168.1783752, 141.6435089, -293.5567017, 296.2409973
8: -185.4225922, 127.8424225, -204.8636932, 141.1464844, -326.5690918, 332.7061157
9: -138.4196777, 137.0056458, -153.2740479, 151.4264526, -289.8460999, 290.2796936

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
time: 8.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
time: 8.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -151.8849182, 121.8413544, -169.9797363, 136.2014008, -288.0863037, 291.8211060
1: -127.8988647, 107.4704132, -143.0194855, 120.0864716, -247.9853058, 250.4898682
2: -167.6065369, 109.4788895, -187.5039520, 122.4066772, -290.0132141, 296.9828186
3: -177.2960968, 93.9522781, -198.2240753, 104.9546509, -282.2507324, 292.1763000
4: -163.6575317, 124.9431000, -182.8902588, 139.7409058, -303.3983765, 307.8333740
5: -146.2171478, 113.9130402, -163.5690002, 127.5075226, -273.7246399, 277.4820557
6: -140.1015472, 134.8249817, -156.5221100, 150.6459808, -290.7475281, 291.3470764
7: -151.9131775, 128.0626221, -170.0911102, 143.2683563, -295.1815186, 298.1536865
8: -185.4225922, 127.8424225, -207.2163239, 142.7386475, -328.1612549, 335.0586853
9: -138.4196777, 137.0056458, -155.0163879, 153.1200867, -291.5397339, 292.0220032

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
time: 7.22 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
time: 7.33 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -155.2188721, 124.4811859, -172.3191528, 138.1140442, -293.3329163, 296.8003540
1: -130.6434479, 109.7616348, -145.1225433, 121.8587799, -252.5022278, 254.8841858
2: -171.2512054, 111.8447571, -190.1643219, 124.1976624, -295.4488525, 302.0090942
3: -181.1500702, 95.9411545, -201.0113373, 106.5129166, -287.6629944, 296.9524841
4: -167.1795349, 127.6565628, -185.5228424, 141.7827606, -308.9622803, 313.1794128
5: -149.4000397, 116.4044952, -165.8747253, 129.3693085, -278.7693481, 282.2791748
6: -143.1104889, 137.7032623, -158.6717529, 152.8284760, -295.9389648, 296.3749390
7: -155.2337036, 130.8485718, -172.5426941, 145.3219452, -300.5556335, 303.3912354
8: -189.4050903, 130.5517273, -210.1320801, 144.9095459, -334.3146362, 340.6837769
9: -141.4439697, 139.9403076, -157.2361145, 155.3870239, -296.8309326, 297.1764221

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1779742, upper bound: 315.1773191
time: 7.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1779742, upper bound: 315.1773191
time: 6.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.11 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1844561, upper bound: 315.1827275
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1844561, upper bound: 315.1828680
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1845656, upper bound: 315.1836369
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1845656, upper bound: 315.1842940
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1820620, upper bound: 315.1820620
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1820620, upper bound: 315.1821880
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1821880, upper bound: 315.1829780
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1821880, upper bound: 315.1835861
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1815256, upper bound: 315.1797149
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1829920, upper bound: 315.1803887
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1815256, upper bound: 315.1797149
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1829920, upper bound: 315.1803887
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1791146, upper bound: 315.1781991
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1791146, upper bound: 315.1782833
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1792239, upper bound: 315.1790194
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1792239, upper bound: 315.1795233
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1826092, upper bound: 315.1826092
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1826092, upper bound: 315.1827907
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1827907, upper bound: 315.1836072
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1827907, upper bound: 315.1844264
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1781545, upper bound: 315.1797806
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1781545, upper bound: 315.1798912
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1783287, upper bound: 315.1808458
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1783287, upper bound: 315.1814284
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1773191, upper bound: 315.1779742
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1779742, upper bound: 315.1773191
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.11
Output dim: 6, lower bound: -315.1779742, upper bound: 315.1773191
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.11
Output dim: 6, lower bound: -315.1779742, upper bound: 315.1785834
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=317.01361083984375
rel_dist={6: [-315.19656872135204, 315.19656872135204]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1874325, upper bound: 315.1868042
time: 8.02 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1863413, upper bound: 315.1863413
time: 8.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.37
Output dim: 6, lower bound: -315.1874325, upper bound: 315.1868042
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.37
Output dim: 6, lower bound: -315.1863413, upper bound: 315.1863413

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -159.8907318, 128.2099457, -171.3649292, 137.3573456, -297.2480774, 299.5748596
1: -134.5814209, 113.0495148, -144.1749115, 121.0912247, -255.6726227, 257.2244263
2: -176.4401398, 115.2283859, -189.0219269, 123.4378433, -299.8779602, 304.2503052
3: -186.6460419, 98.8347626, -199.9467010, 105.8035812, -292.4496155, 298.7814331
4: -172.2470398, 131.5870361, -184.4737854, 140.9761505, -313.2232056, 316.0608215
5: -153.8958435, 119.9840393, -164.9434052, 128.6336517, -282.5294800, 284.9274292
6: -147.3485260, 141.8417816, -157.7695923, 151.8925323, -299.2410583, 299.6113892
7: -160.0037842, 134.8331451, -171.4868622, 144.4828339, -304.4866028, 306.3200073
8: -195.0727539, 134.4256897, -208.8616943, 143.9795074, -339.0522461, 343.2873840
9: -145.7881622, 144.1698914, -156.2413025, 154.4283905, -300.2165527, 300.4111938

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1830146, upper bound: 315.1835684
time: 13.18 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1822625, upper bound: 315.1817307
time: 12.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -161.3736572, 129.3796387, -166.5570679, 133.5321655, -294.9057617, 295.9367065
1: -135.7423553, 114.0286865, -140.1166382, 117.6922989, -253.4346619, 254.1453247
2: -177.9850922, 116.2186508, -183.7142792, 119.9934158, -297.9785156, 299.9329224
3: -188.2836304, 99.6429520, -194.3041077, 102.8299637, -291.1135864, 293.9470520
4: -173.7100220, 132.6823883, -179.2826843, 136.9952850, -310.7052917, 311.9650269
5: -155.3129730, 121.0364609, -160.3127289, 125.0132523, -280.3261719, 281.3491821
6: -148.6860504, 143.0573425, -153.3558044, 147.6227417, -296.3086853, 296.4131470
7: -161.3879852, 136.0165405, -166.6399078, 140.4224396, -301.8104248, 302.6564331
8: -196.7814026, 135.5856171, -203.0482483, 139.9822388, -336.7636414, 338.6338501
9: -147.0498047, 145.3934479, -151.8244781, 150.0858154, -297.1356201, 297.2179260

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1816961, upper bound: 315.1828794
time: 11.86 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1811658, upper bound: 315.1811658
time: 11.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.66 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.66
Output dim: 6, lower bound: -315.1830146, upper bound: 315.1835684
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.66
Output dim: 6, lower bound: -315.1822625, upper bound: 315.1817307
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.66
Output dim: 6, lower bound: -315.1816961, upper bound: 315.1828794
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.66
Output dim: 6, lower bound: -315.1811658, upper bound: 315.1811658

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -158.7499390, 127.3063049, -168.3036804, 134.9329224, -293.6827698, 295.6099854
1: -133.6279602, 112.2484131, -141.6151581, 118.9419174, -252.5698853, 253.8635712
2: -175.1867676, 114.4200897, -185.6585693, 121.2691956, -296.4559326, 300.0786438
3: -185.3063660, 98.1365891, -196.3520508, 103.9293137, -289.2356873, 294.4886169
4: -171.0211029, 130.6559906, -181.1854248, 138.4786682, -309.4997559, 311.8414001
5: -152.7970581, 119.1364059, -161.9946747, 126.3594437, -279.1564941, 281.1310730
6: -146.2947388, 140.8326416, -154.9415131, 149.1837311, -295.4784546, 295.7741699
7: -158.8660889, 133.8767395, -168.4339600, 141.9162750, -300.7823181, 302.3106995
8: -193.7009277, 133.4940796, -205.1804199, 141.4811401, -335.1820679, 338.6744995
9: -144.7524414, 143.1533356, -153.4624939, 151.6993256, -296.4517212, 296.6158142

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1788564, upper bound: 315.1800459
time: 10.05 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1801450, upper bound: 315.1809019
time: 9.46 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -158.8609161, 127.3934250, -181.0204315, 145.0110779, -303.8719788, 308.4138489
1: -133.7203369, 112.3270721, -152.3089142, 127.8651962, -261.5855408, 264.6359863
2: -175.3076172, 114.4982910, -199.6770782, 130.3618927, -305.6694641, 314.1753540
3: -185.4355316, 98.2031555, -211.0937347, 111.7343674, -297.1698914, 309.2968445
4: -171.1401215, 130.7467194, -194.7498016, 148.8835297, -320.0236511, 325.4964600
5: -152.9036560, 119.2178116, -174.2060089, 135.8950958, -288.7987671, 293.4238281
6: -146.3983002, 140.9310760, -166.5470276, 160.3799133, -306.7781982, 307.4780884
7: -158.9750214, 133.9694061, -181.2271881, 152.6086731, -311.5836792, 315.1965942
8: -193.8341370, 133.5828552, -220.5353699, 151.9807129, -345.8148193, 354.1181946
9: -144.8526306, 143.2523041, -165.1540680, 163.0715637, -307.9241943, 308.4063721

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1781882, upper bound: 315.1780910
time: 7.71 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1793907, upper bound: 315.1788518
time: 9.79 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -160.2486877, 128.4886475, -163.5188141, 131.1261292, -291.3748169, 292.0074463
1: -134.8014984, 113.2387085, -137.5758972, 115.5591583, -250.3606567, 250.8146057
2: -176.7491150, 115.4221191, -180.3760223, 117.8415833, -294.5906982, 295.7981262
3: -186.9626617, 98.9540253, -190.7361298, 100.9695740, -287.9322510, 289.6901550
4: -172.5021210, 131.7647095, -176.0194855, 134.5173035, -307.0193481, 307.7841797
5: -154.2295380, 120.2009506, -157.3861847, 122.7561874, -276.9856873, 277.5870972
6: -147.6469879, 142.0618134, -150.5488739, 144.9340820, -292.5809631, 292.6106873
7: -160.2662811, 135.0733948, -163.6101685, 137.8752289, -298.1414795, 298.6835632
8: -195.4286652, 134.6681213, -199.3948822, 137.5038452, -332.9324951, 334.0629272
9: -146.0287323, 144.3905640, -149.0667267, 147.3771057, -293.4058228, 293.4572754

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1776126, upper bound: 315.1793400
time: 10.64 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1789391, upper bound: 315.1802286
time: 10.05 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -160.3708038, 128.5845337, -176.3603973, 141.3029938, -301.6737976, 304.9448853
1: -134.9031219, 113.3249359, -148.3825836, 124.5725327, -259.4756165, 261.7074585
2: -176.8821106, 115.5079956, -194.5373077, 127.0279770, -303.9100647, 310.0452576
3: -187.1044006, 99.0273056, -205.6292877, 108.8544998, -295.9588928, 304.6565857
4: -172.6327209, 131.8639984, -189.7236786, 145.0279388, -317.6606445, 321.5876770
5: -154.3470001, 120.2906647, -169.7148132, 132.3821564, -286.7291565, 290.0054932
6: -147.7607117, 142.1699219, -162.2699890, 156.2461090, -304.0067749, 304.4398804
7: -160.3862305, 135.1754150, -176.5360870, 148.6749268, -309.0611267, 311.7114868
8: -195.5749512, 134.7652588, -214.9124298, 148.1115265, -343.6864624, 349.6776733
9: -146.1388092, 144.4994812, -160.8783875, 158.8673553, -305.0061646, 305.3778381

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1770489, upper bound: 315.1775237
time: 10.95 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1783314, upper bound: 315.1783314
time: 10.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.90 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 6, lower bound: -315.1788564, upper bound: 315.1800459
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 6, lower bound: -315.1801450, upper bound: 315.1809019
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 6, lower bound: -315.1781882, upper bound: 315.1780910
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 6, lower bound: -315.1793907, upper bound: 315.1788518
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 6, lower bound: -315.1776126, upper bound: 315.1793400
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 6, lower bound: -315.1789391, upper bound: 315.1802286
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 6, lower bound: -315.1770489, upper bound: 315.1775237
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 6, lower bound: -315.1783314, upper bound: 315.1783314

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -149.2503510, 119.7631912, -164.1887360, 131.6629639, -280.9132690, 283.9519348
1: -125.7784271, 105.6864700, -138.2111511, 116.0950699, -241.8735046, 243.8976135
2: -164.7975311, 107.6771240, -181.1566772, 118.3476410, -283.1451111, 288.8338013
3: -174.3154907, 92.4412384, -191.5861816, 101.4616776, -275.7771606, 284.0274048
4: -160.9633789, 122.9119949, -176.8256989, 135.1218567, -296.0851746, 299.7377014
5: -143.6941681, 112.0050735, -158.0487518, 123.2671432, -266.9613037, 270.0538330
6: -137.7044373, 132.5919495, -151.2181702, 145.6132507, -283.3176880, 283.8100281
7: -149.3870087, 125.9159241, -164.3250427, 138.4667358, -287.8536987, 290.2409363
8: -182.3319550, 125.7461624, -200.2532959, 138.1241760, -320.4561157, 325.9994507
9: -136.1141510, 134.7569733, -149.7196198, 148.0613556, -284.1755066, 284.4765930

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1788564, upper bound: 315.1800459
time: 9.38 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1788564, upper bound: 315.1800459
time: 9.37 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -152.9591370, 122.6947556, -166.1626892, 133.2291412, -286.1882935, 288.8573914
1: -128.8308411, 108.2356644, -139.8425140, 117.4592056, -246.2900238, 248.0781708
2: -168.8523712, 110.3047256, -183.3176117, 119.7486420, -288.6010132, 293.6223145
3: -178.5982666, 94.6544113, -193.8728333, 102.6430817, -281.2412720, 288.5271912
4: -164.8753662, 125.9248276, -178.9141541, 136.7304688, -301.6058044, 304.8389893
5: -147.2340393, 114.7783890, -159.9391632, 124.7490692, -271.9830933, 274.7175598
6: -141.0517731, 135.7961578, -153.0039673, 147.3230133, -288.3747864, 288.8001099
7: -153.0732574, 129.0132294, -166.2937317, 140.1197052, -293.1929626, 295.3069458
8: -186.7603302, 128.7577667, -202.6158905, 139.7307281, -326.4910583, 331.3736267
9: -139.4752655, 138.0226135, -151.5125885, 149.8036041, -289.2788696, 289.5352173

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1801450, upper bound: 315.1809019
time: 7.76 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1801450, upper bound: 315.1809019
time: 8.53 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -149.3383789, 119.8319778, -176.8850403, 141.7275391, -291.0657959, 296.7169800
1: -125.8515015, 105.7489243, -148.8898010, 125.0062332, -250.8577271, 254.6387177
2: -164.8928986, 107.7389832, -195.1557159, 127.4276581, -292.3205566, 302.8946838
3: -174.4175262, 92.4936752, -206.3072357, 109.2556610, -283.6731262, 298.8009033
4: -161.0576782, 122.9837418, -190.3698883, 145.5122375, -306.5698853, 313.3536072
5: -143.7785950, 112.0693130, -170.2428436, 132.7887115, -276.5672607, 282.3121643
6: -137.7868805, 132.6699829, -162.8071136, 156.7929382, -294.5797729, 295.4771118
7: -149.4727325, 125.9892578, -177.1005249, 149.1444702, -298.6171875, 303.0897217
8: -182.4371948, 125.8160629, -215.5872040, 148.6101837, -331.0473633, 341.4032288
9: -136.1934204, 134.8353271, -161.3947754, 159.4186554, -295.6119995, 296.2300415

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1781882, upper bound: 315.1780910
time: 8.50 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1781882, upper bound: 315.1780910
time: 8.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -153.0646515, 122.7773743, -178.8782501, 143.3049622, -296.3695984, 301.6556091
1: -128.9184723, 108.3103027, -150.5341187, 126.3808289, -255.2993011, 258.8443909
2: -168.9670410, 110.3789139, -197.3327789, 128.8388367, -297.8058472, 307.7117004
3: -178.7207336, 94.7174606, -208.6114807, 110.4463654, -289.1670837, 303.3289490
4: -164.9884796, 126.0110168, -192.4763794, 147.1329346, -312.1213684, 318.4873657
5: -147.3352966, 114.8555527, -172.1479340, 134.2834320, -281.6187134, 287.0034790
6: -141.1500244, 135.8896637, -164.6066895, 158.5173492, -299.6673584, 300.4963379
7: -153.1765137, 129.1011963, -179.0841980, 150.8097534, -303.9862671, 308.1853638
8: -186.8868561, 128.8419189, -217.9686890, 150.2276459, -337.1145020, 346.8106079
9: -139.5704041, 138.1165466, -163.2023315, 161.1731415, -300.7435303, 301.3188477

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1793907, upper bound: 315.1788518
time: 9.44 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1793907, upper bound: 315.1788518
time: 9.43 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -151.0006866, 121.1411285, -159.4241180, 127.8716965, -278.8723755, 280.5652161
1: -127.1597290, 106.8493652, -134.1880341, 112.7259521, -239.8856506, 241.0373993
2: -166.6351929, 108.8528671, -175.8958435, 114.9333801, -281.5685730, 284.7487183
3: -176.2580872, 93.4112854, -185.9933929, 98.5142975, -274.7723999, 279.4046326
4: -162.7077484, 124.2215652, -171.6806946, 131.1753845, -293.8830872, 295.9022217
5: -145.3656616, 113.2566147, -153.4600067, 119.6795273, -265.0451355, 266.7166138
6: -139.2845459, 134.0427551, -146.8432312, 141.3807678, -280.6653137, 280.8859863
7: -151.0319519, 127.3215256, -159.5201416, 134.4423676, -285.4742737, 286.8416138
8: -184.3594055, 127.1212387, -194.4908295, 134.1619110, -318.5213013, 321.6120300
9: -137.6172180, 136.2175293, -145.3415833, 143.7567444, -281.3739319, 281.5591125

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1743591, upper bound: 315.1767946
time: 7.84 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1738929, upper bound: 315.1756438
time: 9.54 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -154.3291473, 123.7766876, -161.3680115, 129.4146576, -283.7438049, 285.1446533
1: -129.8994446, 109.1369553, -135.7955475, 114.0698471, -243.9692993, 244.9324951
2: -170.2738953, 111.2150040, -178.0239563, 116.3134079, -286.5872498, 289.2389526
3: -180.1059418, 95.3966293, -188.2453308, 99.6773376, -279.7832642, 283.6419678
4: -166.2243347, 126.9309235, -173.7382355, 132.7614899, -298.9857483, 300.6691589
5: -148.5430908, 115.7440414, -155.3215637, 121.1385574, -269.6816406, 271.0655823
6: -142.2885895, 136.9159088, -148.6021118, 143.0649719, -285.3535461, 285.5180054
7: -154.3470306, 130.1026917, -161.4600830, 136.0700989, -290.4171143, 291.5627747
8: -188.3353729, 129.8265686, -196.8187256, 135.7458496, -324.0812073, 326.6452942
9: -140.6366425, 139.1470795, -147.1078644, 145.4727325, -286.1093445, 286.2548828

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1758708, upper bound: 315.1778183
time: 8.55 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1753310, upper bound: 315.1765188
time: 9.45 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -151.0987396, 121.2178268, -172.2436676, 138.0331726, -289.1318970, 293.4614868
1: -127.2411346, 106.9187012, -144.9772034, 121.7258072, -248.9669037, 251.8959045
2: -166.7416687, 108.9216309, -190.0349274, 124.1066895, -290.8483276, 298.9565430
3: -176.3716736, 93.4697647, -200.8617706, 106.3864059, -282.7580261, 294.3315125
4: -162.8125000, 124.3011780, -185.3614349, 141.6704559, -304.4829712, 309.6625977
5: -145.4597931, 113.3283386, -165.7684174, 129.2908173, -274.7506104, 279.0967407
6: -139.3759918, 134.1294861, -158.5457611, 152.6737823, -292.0497742, 292.6752319
7: -151.1276398, 127.4031982, -172.4251709, 145.2257080, -296.3533325, 299.8283691
8: -184.4765930, 127.1988525, -209.9849243, 144.7542572, -329.2308350, 337.1836853
9: -137.7054138, 136.3049011, -157.1342468, 155.2294617, -292.9348755, 293.4391479

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1737265, upper bound: 315.1747810
time: 7.89 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1732947, upper bound: 315.1737557
time: 9.93 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -154.4445648, 123.8671799, -174.2103729, 139.5909576, -294.0355225, 298.0775452
1: -129.9953308, 109.2183304, -146.6016083, 123.0827255, -253.0780487, 255.8199463
2: -170.3995056, 111.2961121, -192.1849976, 125.4992065, -295.8987122, 303.4810181
3: -180.2395630, 95.4656754, -203.1382294, 107.5623627, -287.8018494, 298.6038513
4: -166.3477783, 127.0247192, -187.4422607, 143.2716675, -309.6194458, 314.4669800
5: -148.6541138, 115.8286743, -167.6498413, 130.7640228, -279.4180603, 283.4785156
6: -142.3960266, 137.0180511, -160.3231659, 154.3770294, -296.7730713, 297.3412170
7: -154.4601898, 130.1990051, -174.3860168, 146.8692780, -301.3294678, 304.5850220
8: -188.4736328, 129.9182434, -212.3362732, 146.3525238, -334.8261719, 342.2544250
9: -140.7406006, 139.2500000, -158.9196930, 156.9621582, -297.7027588, 298.1696777

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1752938, upper bound: 315.1758464
time: 8.97 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1747670, upper bound: 315.1747670
time: 8.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.93 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1788564, upper bound: 315.1800459
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1788564, upper bound: 315.1800459
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1801450, upper bound: 315.1809019
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1801450, upper bound: 315.1809019
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1781882, upper bound: 315.1780910
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1781882, upper bound: 315.1780910
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1793907, upper bound: 315.1788518
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1793907, upper bound: 315.1788518
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1743591, upper bound: 315.1767946
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1738929, upper bound: 315.1756438
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1758708, upper bound: 315.1778183
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1753310, upper bound: 315.1765188
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1737265, upper bound: 315.1747810
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1732947, upper bound: 315.1737557
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1752938, upper bound: 315.1758464
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.93
Output dim: 6, lower bound: -315.1747670, upper bound: 315.1747670

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -149.2503510, 119.7631912, -152.7023315, 122.5056763, -271.7559814, 272.4654541
1: -125.7784271, 105.6864700, -128.6064606, 108.0443878, -233.8228149, 234.2929230
2: -164.7975311, 107.6771240, -168.5590820, 110.1282425, -274.9257202, 276.2362061
3: -174.3154907, 92.4412384, -178.2709045, 94.4848175, -268.8002930, 270.7121582
4: -160.9633789, 122.9119949, -164.5859375, 125.7228928, -286.6862183, 287.4979248
5: -143.6941681, 112.0050735, -146.9897766, 114.6085434, -258.3027039, 258.9948425
6: -137.7044373, 132.5919495, -140.7852173, 135.5502014, -273.2546387, 273.3771667
7: -149.3870087, 125.9159241, -152.8285980, 128.8053131, -278.1923218, 278.7445068
8: -182.3319550, 125.7461624, -186.4482422, 128.5603790, -310.8923035, 312.1943970
9: -136.1141510, 134.7569733, -139.2542267, 137.7907715, -273.9049072, 274.0112000

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1758433, upper bound: 315.1765379
time: 8.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1747357, upper bound: 315.1760731
time: 10.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -149.2503510, 119.7631912, -154.2727661, 123.7438583, -272.9942017, 274.0358582
1: -125.7784271, 105.6864700, -129.8379059, 109.0832596, -234.8616943, 235.5243683
2: -164.7975311, 107.6771240, -170.2003326, 111.1809235, -275.9784241, 277.8774414
3: -174.3154907, 92.4412384, -180.0087433, 95.3444824, -269.6599731, 272.4499817
4: -160.9633789, 122.9119949, -166.1425171, 126.8885727, -287.8519287, 289.0544739
5: -143.6941681, 112.0050735, -148.4902802, 115.7278900, -259.4220581, 260.4952698
6: -137.7044373, 132.5919495, -142.2015991, 136.8419495, -274.5463867, 274.7935181
7: -149.3870087, 125.9159241, -154.2991638, 130.0620117, -279.4490051, 280.2150269
8: -182.3319550, 125.7461624, -188.2614441, 129.7923889, -312.1243286, 314.0075684
9: -136.1141510, 134.7569733, -140.5958710, 139.0910797, -275.2052307, 275.3528442

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1758433, upper bound: 315.1765379
time: 9.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1747357, upper bound: 315.1760731
time: 8.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -152.9591370, 122.6947556, -154.7057953, 124.0942383, -277.0533752, 277.4005432
1: -128.8308411, 108.2356644, -130.2629700, 109.4287262, -238.2595673, 238.4986115
2: -168.8523712, 110.3047256, -170.7539673, 111.5509720, -280.4032898, 281.0586853
3: -178.5982666, 94.6544113, -180.5912323, 95.6843948, -274.2826538, 275.2456360
4: -164.8753662, 125.9248276, -166.7059479, 127.3551483, -292.2304382, 292.6307678
5: -147.2340393, 114.7783890, -148.9066925, 116.1120682, -263.3460999, 263.6850891
6: -141.0517731, 135.7961578, -142.5986786, 137.2866516, -278.3384399, 278.3948364
7: -153.0732574, 129.0132294, -154.8275604, 130.4830780, -283.5563354, 283.8407898
8: -186.7603302, 128.7577667, -188.8471985, 130.1908264, -316.9511719, 317.6049194
9: -139.4752655, 138.0226135, -141.0748901, 139.5601196, -279.0354004, 279.0975037

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1775595, upper bound: 315.1776796
time: 10.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1762627, upper bound: 315.1770333
time: 9.26 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -152.9591370, 122.6947556, -156.1769257, 125.2551727, -278.2142639, 278.8716125
1: -128.8308411, 108.2356644, -131.4132080, 110.4000549, -239.2308960, 239.6488647
2: -168.8523712, 110.3047256, -172.2859802, 112.5337448, -281.3860474, 282.5906982
3: -178.5982666, 94.6544113, -182.2158356, 96.4849243, -275.0831604, 276.8702393
4: -164.8753662, 125.9248276, -168.1598663, 128.4423065, -293.3175964, 294.0846863
5: -147.2340393, 114.7783890, -150.3133545, 117.1551666, -264.3892212, 265.0917358
6: -141.0517731, 135.7961578, -143.9259796, 138.4921417, -279.5439148, 279.7221375
7: -153.0732574, 129.0132294, -156.2007599, 131.6566315, -284.7298584, 285.2139893
8: -186.7603302, 128.7577667, -190.5414581, 131.3435974, -318.1039429, 319.2992249
9: -139.4752655, 138.0226135, -142.3264313, 140.7723694, -280.2476196, 280.3490601

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1775595, upper bound: 315.1776796
time: 10.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1762627, upper bound: 315.1770333
time: 10.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -149.3383789, 119.8319778, -165.0496521, 132.2914429, -281.6297607, 284.8816223
1: -125.8515015, 105.7489243, -138.9976654, 116.7152176, -242.5666962, 244.7465820
2: -164.8928986, 107.7389832, -182.1746521, 118.9597244, -283.8526306, 289.9136353
3: -174.4175262, 92.4936752, -192.5952301, 102.0706863, -276.4881592, 285.0888672
4: -161.0576782, 122.9837418, -177.7600403, 135.8287811, -296.8864746, 300.7437134
5: -143.7785950, 112.0693130, -158.8509216, 123.8688660, -267.6474609, 270.9202271
6: -137.7868805, 132.6699829, -152.0604248, 146.4268494, -284.2136841, 284.7304077
7: -149.4727325, 125.9892578, -165.2561798, 139.1901093, -288.6628418, 291.2453918
8: -182.4371948, 125.8160629, -201.3605804, 138.7592316, -321.1964111, 327.1765442
9: -136.1934204, 134.8353271, -150.6121216, 148.8390961, -285.0324402, 285.4473877

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1752605, upper bound: 315.1747516
time: 8.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1740030, upper bound: 315.1741127
time: 8.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -149.3383789, 119.8319778, -167.0842285, 133.9003448, -283.2387085, 286.9161987
1: -125.8515015, 105.7489243, -140.6233521, 118.0835266, -243.9350281, 246.3722534
2: -164.8928986, 107.7389832, -184.3363647, 120.3510361, -285.2439270, 292.0753174
3: -174.4175262, 92.4936752, -194.8706207, 103.2181396, -277.6356201, 287.3642883
4: -161.0576782, 122.9837418, -179.8215942, 137.3788300, -298.4365234, 302.8052368
5: -143.7785950, 112.0693130, -160.7916107, 125.3324509, -269.1110229, 272.8609314
6: -137.7868805, 132.6699829, -153.9015961, 148.1326141, -285.9194946, 286.5715942
7: -149.4727325, 125.9892578, -167.1988525, 140.8410492, -290.3137817, 293.1880798
8: -182.4371948, 125.8160629, -203.7491302, 140.3770599, -322.8141785, 329.5651550
9: -136.1934204, 134.8353271, -152.3828125, 150.5604706, -286.7538452, 287.2180786

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1752605, upper bound: 315.1747516
time: 8.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1740030, upper bound: 315.1741127
time: 8.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -153.0646515, 122.7773743, -167.0798645, 133.8987274, -286.9633789, 289.8572083
1: -128.9184723, 108.3103027, -140.6724701, 118.1152802, -247.0337524, 248.9827728
2: -168.9670410, 110.3789139, -184.3917389, 120.3966064, -289.3636169, 294.7706604
3: -178.7207336, 94.7174606, -194.9426727, 103.2827606, -282.0034790, 289.6601257
4: -164.9884796, 126.0110168, -179.9044952, 137.4785309, -302.4670105, 305.9155273
5: -147.3352966, 114.8555527, -160.7924957, 125.3912964, -272.7265930, 275.6480103
6: -141.1500244, 135.8896637, -153.8924713, 148.1826172, -289.3326111, 289.7821350
7: -153.1765137, 129.1011963, -167.2765503, 140.8866577, -294.0631714, 296.3777466
8: -186.8868561, 128.8419189, -203.7838287, 140.4081421, -327.2949829, 332.6257324
9: -139.5704041, 138.1165466, -152.4526672, 150.6269531, -290.1973572, 290.5692139

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1769251, upper bound: 315.1758572
time: 8.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1755446, upper bound: 315.1751397
time: 7.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -153.0646515, 122.7773743, -169.0223694, 135.4370422, -288.5017090, 291.7997437
1: -128.9184723, 108.3103027, -142.2261353, 119.4216003, -248.3400726, 250.5364380
2: -168.9670410, 110.3789139, -186.4559174, 121.7246017, -290.6915894, 296.8348389
3: -178.7207336, 94.7174606, -197.1149597, 104.3776398, -283.0983887, 291.8324280
4: -164.9884796, 126.0110168, -181.8742828, 138.9581604, -303.9466553, 307.8853149
5: -147.3352966, 114.8555527, -162.6476440, 126.7844238, -274.1197205, 277.5031738
6: -141.1500244, 135.8896637, -155.6546326, 149.8127899, -290.9628296, 291.5443115
7: -153.1765137, 129.1011963, -169.1329346, 142.4627228, -295.6392212, 298.2341003
8: -186.8868561, 128.8419189, -206.0686340, 141.9523468, -328.8392029, 334.9105530
9: -139.5704041, 138.1165466, -154.1430206, 152.2693939, -291.8397827, 292.2594910

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1769251, upper bound: 315.1758572
time: 8.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1755446, upper bound: 315.1751397
time: 7.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -149.5988770, 120.0331802, -144.7463226, 116.1698685, -265.7687378, 264.7795105
1: -125.9818192, 105.8634720, -121.6566620, 102.2951279, -228.2769470, 227.5201263
2: -165.0944519, 107.8613663, -159.5195770, 104.3564377, -269.4508667, 267.3809509
3: -174.6127319, 92.5518494, -168.6418762, 89.3191757, -263.9319153, 261.1937256
4: -161.2078400, 123.0750885, -155.7295227, 119.0185623, -280.2264099, 278.8045959
5: -144.0186157, 112.2127228, -139.2424469, 108.6392975, -252.6579132, 251.4551544
6: -137.9943390, 132.7992706, -133.1842041, 128.1953430, -266.1896973, 265.9834595
7: -149.6340485, 126.1480713, -144.6216736, 122.0247574, -271.6587830, 270.7697449
8: -182.6640167, 125.9699478, -176.5415649, 121.9373550, -304.6013794, 302.5115051
9: -136.3437042, 134.9668427, -131.8287659, 130.4111328, -266.7548218, 266.7955933

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1732512, upper bound: 315.1751980
time: 8.25 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1731030, upper bound: 315.1751733
time: 9.04 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -150.2135773, 120.5224609, -157.5362396, 126.3880005, -276.6015625, 278.0587158
1: -126.5014648, 106.2981262, -132.6079102, 111.4038620, -237.9053345, 238.9060364
2: -165.7721100, 108.2983856, -173.8249969, 113.6027603, -279.3748779, 282.1233826
3: -175.3365326, 92.9295273, -183.7830963, 97.3584137, -272.6949463, 276.7126160
4: -161.8675385, 123.5793381, -169.6643982, 129.6353912, -291.5028687, 293.2437134
5: -144.6119537, 112.6732407, -151.6513062, 118.2802048, -262.8920898, 264.3245544
6: -138.5601349, 133.3473053, -145.1052856, 139.7119446, -278.2720642, 278.4525757
7: -150.2498322, 126.6658707, -157.6447144, 132.8688660, -283.1187134, 284.3105774
8: -183.4060059, 126.4736252, -192.2025757, 132.6082306, -316.0142212, 318.6762085
9: -136.9039459, 135.5174561, -143.6304626, 142.0755615, -278.9794922, 279.1478577

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1730263, upper bound: 315.1743914
time: 8.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1727736, upper bound: 315.1743273
time: 8.99 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -152.9375305, 122.6768112, -146.7983398, 117.8003540, -270.7378540, 269.4751282
1: -128.7302094, 108.1583176, -123.3542480, 103.7137909, -232.4440002, 231.5125427
2: -168.7440338, 110.2306976, -161.7677917, 105.8167877, -274.5608215, 271.9984131
3: -178.4720764, 94.5432281, -171.0178070, 90.5466995, -269.0187378, 265.5610352
4: -164.7350769, 125.7928085, -157.9029388, 120.6945343, -285.4295654, 283.6957397
5: -147.2060089, 114.7076569, -141.2082520, 110.1814194, -257.3874207, 255.9159088
6: -141.0074615, 135.6813965, -135.0412750, 129.9741821, -270.9816284, 270.7226562
7: -152.9589233, 128.9375458, -146.6708984, 123.7443619, -276.7032776, 275.6084595
8: -186.6522827, 128.6841736, -178.9993896, 123.6121902, -310.2644653, 307.6835327
9: -139.3722534, 137.9052887, -133.6966095, 132.2236328, -271.5958862, 271.6018982

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1749624, upper bound: 315.1763229
time: 9.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1748942, upper bound: 315.1763278
time: 8.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -153.5371552, 123.1540451, -159.4759674, 127.9276352, -281.4647827, 282.6300049
1: -129.2371674, 108.5822372, -134.2119751, 112.7449341, -241.9820862, 242.7942200
2: -169.4054718, 110.6569519, -175.9487610, 114.9798355, -284.3852539, 286.6057129
3: -179.1786194, 94.9119263, -186.0301819, 98.5190735, -277.6976929, 280.9421082
4: -165.3788910, 126.2847443, -171.7176514, 131.2182465, -296.5971375, 298.0023499
5: -147.7845154, 115.1568604, -153.5088043, 119.7360001, -267.5205078, 268.6656494
6: -141.5595856, 136.2161102, -146.8605652, 141.3925323, -282.9521179, 283.0766602
7: -153.5600433, 129.4429474, -159.5807190, 134.4932098, -288.0531921, 289.0236511
8: -187.3761597, 129.1750031, -194.5258636, 134.1889496, -321.5651245, 323.7008362
9: -139.9189606, 138.4424744, -145.3930206, 143.7878723, -283.7068176, 283.8354492

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1746556, upper bound: 315.1755109
time: 9.14 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1745149, upper bound: 315.1755026
time: 9.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -149.6967163, 120.1097336, -157.7536469, 126.4807663, -276.1774902, 277.8633728
1: -126.0630951, 105.9326630, -132.6106262, 111.4333038, -237.4963684, 238.5432892
2: -165.2006836, 107.9299698, -173.8712921, 113.6670532, -278.8677368, 281.8012695
3: -174.7260742, 92.6102066, -183.7344971, 97.3161621, -272.0422058, 276.3446960
4: -161.3123474, 123.1545868, -169.6112976, 129.6692200, -290.9815369, 292.7658691
5: -144.1125946, 112.2843323, -151.7428894, 118.3959961, -262.5086060, 264.0272217
6: -138.0855255, 132.8858643, -145.0613708, 139.6577148, -277.7432251, 277.9472351
7: -149.7295837, 126.2295914, -157.7210236, 132.9718933, -282.7014771, 283.9505920
8: -182.7809296, 126.0473785, -192.2580109, 132.6815643, -315.4624939, 318.3053894
9: -136.4317474, 135.0540619, -143.8037109, 142.0514374, -278.4831848, 278.8577881

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1729991, upper bound: 315.1738272
time: 10.16 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1727450, upper bound: 315.1737574
time: 9.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -153.0523376, 122.7668304, -159.8235474, 128.1230621, -281.1754150, 282.5903625
1: -128.8256226, 108.2392883, -134.3212280, 112.8626785, -241.6882629, 242.5605011
2: -168.8689728, 110.3114014, -176.1352997, 115.1358795, -284.0048523, 286.4466858
3: -178.6050262, 94.6119690, -186.1313934, 98.5549164, -277.1599121, 280.7433472
4: -164.8579254, 125.8861313, -171.8032532, 131.3567810, -296.2146912, 297.6893921
5: -147.3164520, 114.7918777, -153.7255402, 119.9504013, -267.2668457, 268.5174255
6: -141.1143341, 135.7830505, -146.9340057, 141.4513397, -282.5656738, 282.7170410
7: -153.0714569, 129.0334167, -159.7861786, 134.7041779, -287.7756348, 288.8195801
8: -186.7898102, 128.7752991, -194.7357483, 134.3664551, -321.1562500, 323.5110168
9: -139.4756775, 138.0077820, -145.6842957, 143.8783112, -283.3540039, 283.6920471

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1746184, upper bound: 315.1748955
time: 9.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1744845, upper bound: 315.1748842
time: 8.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -153.6519165, 123.2440186, -172.2834930, 138.0756683, -291.7275085, 295.5274353
1: -129.3325043, 108.6631699, -144.9880219, 121.7335510, -251.0660553, 253.6511841
2: -169.5303955, 110.7376251, -190.0711060, 124.1409073, -293.6712952, 300.8087158
3: -179.3114777, 94.9805984, -200.8820343, 106.3821487, -285.6936340, 295.8626099
4: -165.5016785, 126.3779984, -185.3838654, 141.6997528, -307.2013550, 311.7618713
5: -147.8948822, 115.2410278, -165.8022461, 129.3355255, -277.2303467, 281.0432739
6: -141.6665192, 136.3177185, -158.5489349, 152.6733246, -294.3398132, 294.8666382
7: -153.6725616, 129.5387421, -172.4714661, 145.2630463, -298.9355164, 302.0101929
8: -187.5136261, 129.2660980, -210.0007782, 144.7665405, -332.2801514, 339.2668762
9: -140.0222931, 138.5448914, -157.1730194, 155.2456360, -295.2679443, 295.7178650

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1742801, upper bound: 315.1741617
time: 16.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1741084, upper bound: 315.1741084
time: 10.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 28.63 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1758433, upper bound: 315.1765379
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1747357, upper bound: 315.1760731
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1758433, upper bound: 315.1765379
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1747357, upper bound: 315.1760731
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1775595, upper bound: 315.1776796
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1762627, upper bound: 315.1770333
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1775595, upper bound: 315.1776796
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1762627, upper bound: 315.1770333
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1752605, upper bound: 315.1747516
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1740030, upper bound: 315.1741127
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1752605, upper bound: 315.1747516
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1740030, upper bound: 315.1741127
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1769251, upper bound: 315.1758572
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1755446, upper bound: 315.1751397
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1769251, upper bound: 315.1758572
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1755446, upper bound: 315.1751397
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1732512, upper bound: 315.1751980
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1731030, upper bound: 315.1751733
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1730263, upper bound: 315.1743914
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1727736, upper bound: 315.1743273
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1749624, upper bound: 315.1763229
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1748942, upper bound: 315.1763278
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1746556, upper bound: 315.1755109
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1745149, upper bound: 315.1755026
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1729991, upper bound: 315.1738272
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1727450, upper bound: 315.1737574
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1746184, upper bound: 315.1748955
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1744845, upper bound: 315.1748842
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1742801, upper bound: 315.1741617
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 28.63
Output dim: 6, lower bound: -315.1741084, upper bound: 315.1741084
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=317.01361083984375
rel_dist={6: [-315.19620904713327, 315.1962090416698]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1863189, upper bound: 315.1860745
time: 11.83 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1859104, upper bound: 315.1859104
time: 9.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.54 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.54
Output dim: 6, lower bound: -315.1863189, upper bound: 315.1860745
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.54
Output dim: 6, lower bound: -315.1859104, upper bound: 315.1859104

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -159.8907318, 128.2099457, -164.2411041, 131.6770325, -291.5677490, 292.4510498
1: -134.5814209, 113.0495148, -138.2200012, 116.0994110, -250.6808319, 251.2695160
2: -176.4401398, 115.2283859, -181.2109375, 118.3401794, -294.7803345, 296.4393311
3: -186.6460419, 98.8347626, -191.6913452, 101.4787903, -288.1248169, 290.5260925
4: -172.2470398, 131.5870361, -176.8845520, 135.1467590, -307.3937988, 308.4715881
5: -153.8958435, 119.9840393, -158.0845032, 123.2616348, -277.1574707, 278.0685425
6: -147.3485260, 141.8417816, -151.3006897, 145.6549225, -293.0034485, 293.1424561
7: -160.0037842, 134.8331451, -164.3582458, 138.4909821, -298.4947510, 299.1914062
8: -195.0727539, 134.4256897, -200.3012238, 138.0483704, -333.1211243, 334.7269287
9: -145.7881622, 144.1698914, -149.7514954, 148.0600281, -293.8482056, 293.9213867

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1812787, upper bound: 315.1815789
time: 13.18 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1809940, upper bound: 315.1807961
time: 11.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -161.3736572, 129.3796387, -160.1626282, 128.4385681, -289.8121948, 289.5422668
1: -135.7423553, 114.0286865, -134.7379150, 113.1862564, -248.9286194, 248.7666016
2: -177.9850922, 116.2186508, -176.6710358, 115.4099350, -293.3950195, 292.8896790
3: -188.2836304, 99.6429520, -186.8376007, 98.9050827, -287.1887207, 286.4805603
4: -173.7100220, 132.6823883, -172.4116211, 131.7266846, -305.4366150, 305.0939636
5: -155.3129730, 121.0364609, -154.1575165, 120.1989746, -275.5119019, 275.1939697
6: -148.6860504, 143.0573425, -147.5080414, 141.9757080, -290.6617126, 290.5653687
7: -161.3879852, 136.0165405, -160.2088928, 135.0283356, -296.4162598, 296.2254333
8: -196.7814026, 135.5856171, -195.3312683, 134.6671143, -331.4485168, 330.9168396
9: -147.0498047, 145.3934479, -145.9699097, 144.3328857, -291.3826904, 291.3633423

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1807959, upper bound: 315.1813076
time: 11.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1805880, upper bound: 315.1805880
time: 11.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.53 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.53
Output dim: 6, lower bound: -315.1812787, upper bound: 315.1815789
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.53
Output dim: 6, lower bound: -315.1809940, upper bound: 315.1807961
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.53
Output dim: 6, lower bound: -315.1807959, upper bound: 315.1813076
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.53
Output dim: 6, lower bound: -315.1805880, upper bound: 315.1805880

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -157.6106567, 126.4038849, -161.1754456, 129.2488708, -286.8595276, 287.5793152
1: -132.6751251, 111.4485397, -135.6565247, 113.9470215, -246.6221466, 247.1050720
2: -173.9350586, 113.6130600, -177.8426361, 116.1683807, -290.1034241, 291.4556885
3: -183.9686432, 97.4390640, -188.0914764, 99.6019821, -283.5706177, 285.5305481
4: -169.7978210, 129.7268372, -173.5918884, 132.6459656, -302.4437866, 303.3186951
5: -151.6994934, 118.2900162, -155.1313019, 120.9840927, -272.6835938, 273.4213257
6: -145.2425232, 139.8245087, -148.4689178, 142.9422607, -288.1847839, 288.2933655
7: -157.7300720, 132.9212646, -161.3011322, 135.9203949, -293.6504517, 294.2224121
8: -192.3309937, 132.5646973, -196.6148224, 135.5467224, -327.8777161, 329.1795044
9: -143.7183685, 142.1377106, -146.9687958, 145.3271790, -289.0455017, 289.1065063

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1771780, upper bound: 315.1777270
time: 15.50 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1784618, upper bound: 315.1788939
time: 12.17 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -157.9960175, 126.7077255, -173.6726837, 139.1524048, -297.1484375, 300.3804016
1: -132.9969635, 111.7203217, -146.1697998, 122.7190399, -255.7160034, 257.8901062
2: -174.3563538, 113.8850632, -191.6202393, 125.1057892, -299.4621582, 305.5052795
3: -184.4187469, 97.6725235, -202.5846252, 107.2738876, -291.6926270, 300.2571411
4: -170.2105560, 130.0411682, -186.9220886, 142.8722839, -313.0828247, 316.9632263
5: -152.0704193, 118.5742798, -167.1329651, 130.3552094, -282.4256287, 285.7072449
6: -145.6002502, 140.1661530, -159.8760681, 153.9458313, -299.5460510, 300.0421753
7: -158.1109314, 133.2438965, -173.8759613, 146.4294891, -304.5404053, 307.1198730
8: -192.7937164, 132.8751221, -211.7058868, 145.8662415, -338.6599731, 344.5809937
9: -144.0668640, 142.4816132, -158.4617615, 156.5053711, -300.5722351, 300.9433594

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1769493, upper bound: 315.1769346
time: 11.56 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1781792, upper bound: 315.1779783
time: 15.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -159.1199341, 127.5947723, -157.1424103, 126.0468292, -285.1667480, 284.7371521
1: -133.8570557, 112.4461899, -132.2120056, 111.0658951, -244.9229431, 244.6582031
2: -175.5090179, 114.6230545, -173.3523712, 113.2711182, -288.7801514, 287.9754333
3: -185.6372375, 98.2625351, -183.2904510, 97.0556488, -282.6928711, 281.5529785
4: -171.2907867, 130.8444214, -169.1681671, 129.2637482, -300.5545349, 300.0125732
5: -153.1424408, 119.3627548, -151.2480774, 117.9554138, -271.0978394, 270.6108398
6: -146.6045380, 141.0627441, -144.7176819, 139.3027496, -285.9072876, 285.7803955
7: -159.1410370, 134.1269531, -157.1971588, 132.4960022, -291.6370239, 291.3240967
8: -194.0713959, 133.7482147, -191.6994324, 132.2041931, -326.2755737, 325.4476318
9: -145.0043640, 143.3840179, -143.2287292, 141.6400757, -286.6443787, 286.6127319

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1767174, upper bound: 315.1774498
time: 10.88 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1780106, upper bound: 315.1786253
time: 10.52 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -159.5315094, 127.9191132, -170.0312805, 136.2630920, -295.7945862, 297.9503784
1: -134.2004700, 112.7360611, -143.0628510, 120.1162491, -254.3167114, 255.7989197
2: -175.9589386, 114.9132614, -187.5704041, 122.4933090, -298.4522400, 302.4836121
3: -186.1174622, 98.5117950, -198.2430267, 104.9729614, -291.0904236, 296.7547302
4: -171.7315979, 131.1795349, -182.9275513, 139.8149261, -311.5465088, 314.1070862
5: -153.5384979, 119.6664658, -163.6234283, 127.6172485, -281.1557617, 283.2898865
6: -146.9863892, 141.4271240, -156.4857178, 150.6594238, -297.6458130, 297.9128113
7: -159.5478058, 134.4712524, -170.1739960, 143.3391266, -302.8869324, 304.6452332
8: -194.5652618, 134.0791779, -207.2796631, 142.8498840, -337.4151306, 341.3588257
9: -145.3764343, 143.7511139, -155.0864258, 153.1765137, -298.5529480, 298.8374939

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1765389, upper bound: 315.1767330
time: 10.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1777968, upper bound: 315.1777968
time: 10.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 6, lower bound: -315.1771780, upper bound: 315.1777270
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 6, lower bound: -315.1784618, upper bound: 315.1788939
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 6, lower bound: -315.1769493, upper bound: 315.1769346
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 6, lower bound: -315.1781792, upper bound: 315.1779783
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 6, lower bound: -315.1767174, upper bound: 315.1774498
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 6, lower bound: -315.1780106, upper bound: 315.1786253
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 6, lower bound: -315.1765389, upper bound: 315.1767330
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 6, lower bound: -315.1777968, upper bound: 315.1777968

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -148.1116028, 118.8611298, -153.8248596, 123.4071503, -271.5187378, 272.6859741
1: -124.8266296, 104.8870087, -129.5734406, 108.8602600, -233.6868744, 234.4604492
2: -163.5464325, 106.8705826, -169.7986603, 110.9473419, -274.4937134, 276.6692200
3: -172.9783783, 91.7443771, -179.5771027, 95.1911774, -268.1695557, 271.3214111
4: -159.7399902, 121.9830170, -165.8018341, 126.6474991, -286.3874817, 287.7848511
5: -142.5971680, 111.1591339, -148.0825500, 115.4601974, -258.0573730, 259.2416687
6: -136.6528625, 131.5846558, -141.8146820, 136.5605774, -273.2134399, 273.3992920
7: -148.2514801, 124.9612732, -153.9586487, 129.7567139, -278.0081482, 278.9199219
8: -180.9628296, 124.8167648, -187.8093567, 129.5479584, -310.5107727, 312.6260681
9: -135.0804443, 133.7422943, -140.2804260, 138.8251343, -273.9055786, 274.0227051

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1733271, upper bound: 315.1741338
time: 10.43 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1732312, upper bound: 315.1737959
time: 11.60 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -151.8193970, 121.7920685, -156.8434753, 125.7999115, -277.6192932, 278.6355286
1: -127.8778458, 107.4356537, -132.0691833, 110.9458542, -238.8237000, 239.5048370
2: -167.6001587, 109.4975510, -173.1055298, 113.0916519, -280.6918030, 282.6030579
3: -177.2602081, 93.9566727, -183.0739899, 96.9982986, -274.2584839, 277.0306702
4: -163.6514740, 124.9952774, -168.9952545, 129.1080933, -292.7595825, 293.9905396
5: -146.1359863, 113.9318771, -150.9699554, 117.7250366, -263.8610229, 264.9017944
6: -139.9992828, 134.7876129, -144.5485992, 139.1752625, -279.1745605, 279.3362122
7: -151.9368896, 128.0574799, -156.9701691, 132.2841949, -284.2210693, 285.0276184
8: -185.3898621, 127.8279266, -191.4247894, 132.0040741, -317.3939209, 319.2527161
9: -138.4407501, 137.0067444, -143.0226440, 141.4905090, -279.9312744, 280.0293579

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1749666, upper bound: 315.1756759
time: 11.66 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1748418, upper bound: 315.1752066
time: 12.29 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -148.4558868, 119.1321411, -166.2854614, 133.2868195, -281.7427063, 285.4176025
1: -125.1138458, 105.1298141, -140.0606384, 117.6110840, -242.7249298, 245.1904449
2: -163.9221497, 107.1132431, -183.5422668, 119.8635330, -283.7856750, 290.6554565
3: -173.3800812, 91.9525223, -194.0324707, 102.8450928, -276.2251587, 285.9849854
4: -160.1086884, 122.2633896, -179.0974731, 136.8488464, -296.9575195, 301.3608704
5: -142.9283142, 111.4126434, -160.0522919, 124.8060379, -267.7343445, 271.4649353
6: -136.9726868, 131.8896942, -153.1947632, 147.5366821, -284.5093689, 285.0844116
7: -148.5908966, 125.2490921, -166.5027771, 140.2405853, -288.8314819, 291.7518311
8: -181.3755188, 125.0934296, -202.8661652, 139.8433380, -321.2188721, 327.9595642
9: -135.3914642, 134.0490570, -151.7450714, 149.9783325, -285.3698120, 285.7941284

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1730725, upper bound: 315.1733741
time: 11.69 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1729673, upper bound: 315.1730124
time: 13.27 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -152.1938324, 122.0869217, -169.3382263, 135.6997375, -287.8935547, 291.4251404
1: -128.1902466, 107.6994400, -142.5782623, 119.7148819, -247.9051056, 250.2777100
2: -168.0091858, 109.7615128, -186.8759460, 122.0235977, -290.0327759, 296.6374512
3: -177.6970062, 94.1831589, -197.5621490, 104.6667023, -282.3637085, 291.7453003
4: -164.0523987, 125.3005447, -182.3207092, 139.3292236, -303.3816223, 307.6212463
5: -146.4962006, 114.2077103, -162.9684448, 127.0934677, -273.5895996, 277.1760864
6: -140.3465424, 135.1194305, -155.9491425, 150.1750946, -290.5216370, 291.0685425
7: -152.3065033, 128.3706818, -169.5393524, 142.7887726, -295.0952454, 297.9100342
8: -185.8393097, 128.1291656, -206.5095673, 142.3197021, -328.1589966, 334.6387329
9: -138.7791901, 137.3404999, -154.5112305, 152.6634369, -291.4425964, 291.8517456

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1746742, upper bound: 315.1748640
time: 14.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1745578, upper bound: 315.1743828
time: 13.86 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -149.8842773, 120.2568741, -149.8448334, 120.2458115, -270.1300659, 270.1017151
1: -126.2257309, 106.0656128, -126.1729813, 106.0156860, -232.2414246, 232.2385864
2: -165.4084930, 108.0626602, -165.3662720, 108.0860291, -273.4945068, 273.4289246
3: -174.9472504, 92.7275009, -174.8382416, 92.6782303, -267.6254883, 267.5657043
4: -161.5097046, 123.3113022, -161.4334564, 123.3067017, -284.8163452, 284.7447510
5: -144.2903290, 112.4276428, -144.2512054, 112.4718246, -256.7620850, 256.6788330
6: -138.2535858, 133.0548096, -138.1125488, 132.9682617, -271.2218018, 271.1673279
7: -149.9190216, 126.3853531, -149.9058685, 126.3767700, -276.2957458, 276.2911987
8: -183.0170593, 126.2112808, -182.9574280, 126.2459717, -309.2630310, 309.1687012
9: -136.6040955, 135.2220764, -136.5881042, 135.1865540, -271.7906494, 271.8101807

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 62

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1731099, upper bound: 315.1740766
time: 11.31 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1729348, upper bound: 315.1736479
time: 11.15 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -153.2017365, 122.8838730, -152.7363129, 122.5406418, -275.7423706, 275.6201172
1: -128.9562988, 108.3456345, -128.5643158, 108.0143051, -236.9706116, 236.9099274
2: -169.0352478, 110.4170227, -168.5337372, 110.1407166, -279.1759644, 278.9507446
3: -178.7824097, 94.7060242, -178.1883087, 94.4081955, -273.1905518, 272.8942871
4: -165.0143585, 126.0117111, -164.4954681, 125.6663666, -290.6807251, 290.5071716
5: -147.4571838, 114.9069672, -147.0165405, 114.6404800, -262.0976562, 261.9235229
6: -141.2475281, 135.9179993, -140.7294159, 135.4728851, -276.7203369, 276.6473999
7: -153.2231598, 129.1573639, -152.7926025, 128.7975464, -282.0206909, 281.9499207
8: -186.9797668, 128.9076233, -186.4216461, 128.6020355, -315.5817566, 315.3292236
9: -139.6136017, 138.1417999, -139.2164001, 137.7380829, -277.3516846, 277.3581543

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1746558, upper bound: 315.1755297
time: 12.02 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1744351, upper bound: 315.1749732
time: 15.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -150.2509766, 120.5455017, -162.6832275, 130.4252167, -280.6760559, 283.2286682
1: -126.5314178, 106.3239594, -136.9829102, 115.0345612, -241.5659485, 243.3068695
2: -165.8090210, 108.3208923, -179.5322723, 117.2771225, -283.0861206, 287.8531494
3: -175.3748779, 92.9490814, -189.7332306, 100.5666885, -275.9415588, 282.6823120
4: -161.9022675, 123.6096573, -175.1401367, 133.8200836, -295.7223206, 298.7497864
5: -144.6429596, 112.6978531, -156.5775299, 122.0985718, -266.7415161, 269.2753906
6: -138.5938110, 133.3792572, -149.8356628, 144.2821503, -282.8759766, 283.2149048
7: -150.2807007, 126.6917725, -162.8344421, 137.1802216, -287.4609375, 289.5262146
8: -183.4566040, 126.5057144, -198.4806366, 136.8568268, -320.3133545, 324.9863281
9: -136.9352722, 135.5488892, -148.4019623, 146.6816406, -283.6169128, 283.9508667

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1728712, upper bound: 315.1732993
time: 13.40 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1727200, upper bound: 315.1728986
time: 10.54 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -153.5997772, 123.1973419, -165.6228180, 132.7528687, -286.3526306, 288.8201599
1: -129.2880859, 108.6256561, -139.4119415, 117.0618744, -246.3499603, 248.0375977
2: -169.4701996, 110.6975250, -182.7473145, 119.3587112, -288.8289185, 293.4448242
3: -179.2462158, 94.9467239, -193.1360474, 102.3234253, -281.5696411, 288.0827637
4: -165.4407349, 126.3357010, -178.2510376, 136.2144165, -301.6551208, 304.5867004
5: -147.8402405, 115.2004471, -159.3895111, 124.2983932, -272.1386108, 274.5899658
6: -141.6166687, 136.2704010, -152.4945068, 146.8281250, -288.4447937, 288.7648926
7: -153.6162720, 129.4902344, -165.7661285, 139.6367035, -293.2529602, 295.2562866
8: -187.4573669, 129.2274323, -201.9971619, 139.2430115, -326.7003784, 331.2245789
9: -139.9732361, 138.4967041, -151.0708771, 149.2708893, -289.2440491, 289.5675659

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1744377, upper bound: 315.1747739
time: 10.39 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1742414, upper bound: 315.1742414
time: 11.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.05 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1733271, upper bound: 315.1741338
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1732312, upper bound: 315.1737959
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1749666, upper bound: 315.1756759
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1748418, upper bound: 315.1752066
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1730725, upper bound: 315.1733741
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1729673, upper bound: 315.1730124
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1746742, upper bound: 315.1748640
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1745578, upper bound: 315.1743828
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1731099, upper bound: 315.1740766
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1729348, upper bound: 315.1736479
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1746558, upper bound: 315.1755297
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1744351, upper bound: 315.1749732
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1728712, upper bound: 315.1732993
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1727200, upper bound: 315.1728986
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1744377, upper bound: 315.1747739
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 6, lower bound: -315.1742414, upper bound: 315.1742414

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -149.3944244, 119.8758774, -141.3078766, 113.4096375, -262.8040771, 261.1837463
1: -125.8395462, 105.7299347, -118.7998047, 99.9035873, -225.7431335, 224.5297241
2: -164.9336090, 107.7814865, -155.7708130, 101.8891449, -266.8227539, 263.5523071
3: -174.4124451, 92.4691772, -164.7063599, 87.2583771, -261.6707458, 257.1755371
4: -161.0549774, 123.0127411, -152.1057281, 116.2321472, -277.2870789, 275.1184692
5: -143.8052673, 112.1263046, -135.9210510, 106.0348129, -249.8400574, 248.0473480
6: -137.7641907, 132.6358643, -130.0886993, 125.2159195, -262.9801025, 262.7245483
7: -149.5190582, 126.0260773, -141.1974945, 119.1345367, -268.6535950, 267.2235107
8: -182.4562073, 125.8371582, -172.4162140, 119.0413284, -301.4974976, 298.2533569
9: -136.2374115, 134.8412781, -128.7159119, 127.3582535, -263.5956421, 263.5571594

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1742725, upper bound: 315.1747532
time: 12.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1742610, upper bound: 315.1747659
time: 12.51 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -150.3633423, 120.6484222, -154.9404449, 124.3046722, -274.6679993, 275.5888367
1: -126.6600189, 106.4163589, -130.4767456, 109.6134338, -236.2734528, 236.8930969
2: -166.0037079, 108.4718170, -171.0185547, 111.7506332, -277.7543335, 279.4903564
3: -175.5560760, 93.0659332, -180.8464508, 95.8334427, -271.3894958, 273.9122620
4: -162.0974731, 123.8083038, -166.9634705, 127.5561981, -289.6536255, 290.7717285
5: -144.7417145, 112.8528595, -149.1468811, 116.3145218, -261.0562439, 261.9997559
6: -138.6601257, 133.5011597, -142.7974548, 137.4934998, -276.1535645, 276.2986145
7: -150.4910278, 126.8449249, -155.0801544, 130.6984253, -281.1894226, 281.9250793
8: -183.6266174, 126.6303558, -189.1188812, 130.4384308, -314.0650330, 315.7492371
9: -137.1218109, 135.7110443, -141.2980652, 139.7960358, -276.9177856, 277.0090942

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1741883, upper bound: 315.1744557
time: 11.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1741670, upper bound: 315.1744557
time: 12.25 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -149.7673187, 120.1695938, -153.9354401, 123.4106750, -273.1779175, 274.1049805
1: -126.1504593, 105.9926300, -129.4305115, 108.7723236, -234.9227600, 235.4231262
2: -165.3408051, 108.0442886, -169.6934052, 110.9197388, -276.2605591, 277.7377014
3: -174.8473511, 92.6948166, -179.3573151, 95.0229721, -269.8703308, 272.0521240
4: -161.4540863, 123.3168335, -165.5763092, 126.5634613, -288.0175476, 288.8931274
5: -144.1639862, 112.4011307, -148.0523529, 115.5029984, -259.6669922, 260.4534912
6: -138.1100311, 132.9662781, -141.6164398, 136.3395233, -274.4495544, 274.5826416
7: -149.8871918, 126.3379364, -153.9064026, 129.7539825, -279.6411133, 280.2443237
8: -182.9038544, 126.1371613, -187.6647186, 129.4654541, -312.3692932, 313.8018799
9: -136.5743103, 135.1737061, -140.3331146, 138.6464691, -275.2207336, 275.5068054

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1740759, upper bound: 315.1740813
time: 10.49 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1740451, upper bound: 315.1740795
time: 11.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -150.7422485, 120.9467239, -167.3982849, 134.1750183, -284.9172363, 288.3449707
1: -126.9760132, 106.6833801, -140.9544983, 118.3571014, -245.3330994, 247.6378784
2: -166.4175873, 108.7389755, -184.7485962, 120.6565933, -287.0741577, 293.4875793
3: -175.9981689, 93.2951736, -195.2913666, 103.4789581, -279.4771118, 288.5865479
4: -162.5030212, 124.1173325, -180.2493591, 137.7472992, -300.2502747, 304.3666992
5: -145.1061401, 113.1321411, -161.1090088, 125.6556931, -270.7618408, 274.2411499
6: -139.0115509, 133.8369598, -154.1640625, 148.4605713, -287.4721069, 288.0010071
7: -150.8651428, 127.1619263, -167.6126404, 141.1723175, -292.0374756, 294.7744751
8: -184.0814209, 126.9351883, -204.1591949, 140.7235870, -324.8049927, 331.0943604
9: -137.4642944, 136.0487976, -152.7531128, 150.9358826, -288.4001770, 288.8019104

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1565404, upper bound: 315.1557547
time: 12.28 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1549611, upper bound: 315.1551419
time: 11.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -150.6294861, 120.8502502, -138.6614990, 111.3212204, -261.9506836, 259.5117493
1: -126.7938309, 106.5361633, -116.5429993, 98.0064545, -224.8002777, 223.0791473
2: -166.2065125, 108.5967712, -152.8254395, 100.0027466, -266.2092590, 261.4222107
3: -175.7621307, 93.1284027, -161.5415802, 85.5854263, -261.3475647, 254.6699524
4: -162.2605743, 123.9077835, -149.1948090, 114.0085678, -276.2690735, 273.1026001
5: -144.9850922, 112.9910507, -133.3802795, 104.0571976, -249.0422974, 246.3713379
6: -138.8788757, 133.6351929, -127.6251450, 122.8205032, -261.6993713, 261.2603149
7: -150.6569214, 127.0028763, -138.5053864, 116.8926773, -267.5495911, 265.5082703
8: -183.8681030, 126.7952805, -169.1999969, 116.8812027, -300.7492981, 295.9952087
9: -137.2758179, 135.8452759, -126.2616119, 124.9391785, -262.2149963, 262.1068420

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1739263, upper bound: 315.1745691
time: 12.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -315.1739091, upper bound: 315.1745740
time: 11.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -151.6846161, 121.6912842, -150.8444214, 121.0535660, -272.7381897, 272.5356750
1: -127.6875381, 107.2829971, -126.9808960, 106.6891174, -234.3766479, 234.2638855
2: -167.3717194, 109.3479843, -166.4584045, 108.8070679, -276.1787720, 275.8063660
3: -177.0061493, 93.7775879, -175.9733276, 93.2497787, -270.2559204, 269.7508545
4: -163.3948212, 124.7739410, -162.4747772, 124.1228638, -287.5177002, 287.2487183
5: -146.0040894, 113.7821808, -145.2035828, 113.2379150, -259.2420044, 258.9857788
6: -139.8509827, 134.5774994, -138.9875336, 133.8005066, -273.6514587, 273.5649719
7: -151.7157135, 127.8935165, -150.9130096, 127.2205658, -278.9362793, 278.8065186
8: -185.1423035, 127.6593781, -184.1284180, 127.0447998, -312.1871033, 311.7877808
9: -138.2387695, 136.7919464, -137.5014801, 136.0532379, -274.2919922, 274.2934265

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1738009, upper bound: 315.1742029
time: 14.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1737499, upper bound: 315.1742012
time: 13.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -151.0293274, 121.1651993, -151.6643219, 121.6268768, -272.6561890, 272.8295288
1: -127.1269913, 106.8174438, -127.4925461, 107.1437988, -234.2707520, 234.3099670
2: -166.6433868, 108.8784561, -167.1695557, 109.3046341, -275.9480286, 276.0479736
3: -176.2281494, 93.3701706, -176.6332092, 93.5807190, -269.8088379, 270.0033569
4: -162.6886902, 124.2332916, -163.0748901, 124.6540756, -287.3427734, 287.3081665
5: -145.3696747, 113.2859650, -145.8766937, 113.8098526, -259.1795349, 259.1626587
6: -139.2495270, 133.9891510, -139.4985504, 134.2812805, -273.5307617, 273.4877014
7: -151.0518951, 127.3371124, -151.6005554, 127.8356247, -278.8875122, 278.9375610
8: -184.3478241, 127.1165695, -184.9097900, 127.6152802, -311.9631042, 312.0263672
9: -137.6370392, 136.2016907, -138.2325592, 136.5748291, -274.2118530, 274.4342651

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1738009, upper bound: 315.1739644
time: 15.30 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1737487, upper bound: 315.1739629
time: 10.44 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 27.07 seconds
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1742725, upper bound: 315.1747532
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1742610, upper bound: 315.1747659
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1741883, upper bound: 315.1744557
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1741670, upper bound: 315.1744557
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1740759, upper bound: 315.1740813
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1740451, upper bound: 315.1740795
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1565404, upper bound: 315.1557547
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1549611, upper bound: 315.1551419
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1739263, upper bound: 315.1745691
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1739091, upper bound: 315.1745740
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1738009, upper bound: 315.1742029
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1737499, upper bound: 315.1742012
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1738009, upper bound: 315.1739644
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 27.07
Output dim: 6, lower bound: -315.1737487, upper bound: 315.1739629

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -147.0828857, 118.0404129, -139.5900574, 112.0442657, -259.1271362, 257.6304321
1: -123.8937149, 104.0883331, -117.3522797, 98.6824188, -222.5760956, 221.4406128
2: -162.3823700, 106.1480026, -153.8726959, 100.6745377, -263.0569153, 260.0206604
3: -171.7022858, 91.0377350, -162.6893158, 86.1934509, -257.8957520, 253.7270355
4: -158.5195770, 121.0899734, -150.2193146, 114.8003769, -273.3199463, 271.3092957
5: -141.5807190, 110.3953400, -134.2663879, 104.7472839, -246.3280029, 244.6616974
6: -135.6155853, 130.5727081, -128.4905548, 123.6805344, -259.2960815, 259.0632629
7: -147.1967468, 124.0743027, -139.4708862, 117.6826782, -264.8793945, 263.5451965
8: -179.6678467, 123.9309845, -170.3416748, 117.6230927, -297.2909546, 294.2726440
9: -134.1174316, 132.7576294, -127.1400452, 125.8063736, -259.9237671, 259.8976746

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1543108, upper bound: 315.1559505
time: 14.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -315.1538018, upper bound: 315.1545589
time: 10.77 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 26.21 seconds
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 26.21
Output dim: 6, lower bound: -315.1543108, upper bound: 315.1559505
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 26.21
Output dim: 6, lower bound: -315.1538018, upper bound: 315.1545589
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.21
Output dim: 6, lower bound: -315.1742610, upper bound: 315.1747659
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.21
Output dim: 6, lower bound: -315.1739263, upper bound: 315.1745691
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.21
Output dim: 6, lower bound: -315.1739091, upper bound: 315.1745740
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=317.01361083984375
rel_dist={6: [-315.19563397098386, 315.19563397064064]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1835.44 seconds
