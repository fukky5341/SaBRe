## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 587.735384174
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-327.0875244, 260.7777405, -327.0875244, 260.7777405, -587.8651733, 587.8651733)
1: (-275.9752808, 230.9897614, -275.9752808, 230.9897614, -506.9650269, 506.9650269)
2: (-361.3987427, 234.9588470, -361.3987427, 234.9588470, -596.3576050, 596.3576050)
3: (-382.4662781, 202.0105438, -382.4662781, 202.0105438, -584.4767456, 584.4767456)
4: (-352.3798218, 268.6433411, -352.3798218, 268.6433411, -621.0231934, 621.0231934)
5: (-314.9150391, 244.7410431, -314.9150391, 244.7410431, -559.6560059, 559.6560059)
6: (-301.2674561, 290.0480957, -301.2674561, 290.0480957, -591.3155518, 591.3155518)
7: (-328.5839539, 275.4401550, -328.5839539, 275.4401550, -604.0240479, 604.0240479)
8: (-396.7255249, 272.1857910, -396.7255249, 272.1857910, -668.9113159, 668.9113159)
9: (-298.9044800, 294.3448181, -298.9044800, 294.3448181, -593.2492676, 593.2492676)

## BASE Result
execution time: IAR + LP analysis = 1.05 + 12.49 = 13.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -587.7908408, upper bound: 587.7908408


# Binary Search by BASE starts (time budget: 2686.46 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=591.3155517578125
rel_dist={6: [-587.7907620297522, 587.7907620249766]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=591.3155517578125
rel_dist={6: [-587.7906229930563, 587.7906229976039]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=591.3155517578125
rel_dist={6: [-587.7904223681265, 587.7904223711924]}

## Binary Search Result
Binary search time: 50.60 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2635.85 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7730149, upper bound: 587.7624851
time: 11.82 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7560896, upper bound: 587.7560896
time: 7.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.81
Output dim: 6, lower bound: -587.7730149, upper bound: 587.7624851
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.81
Output dim: 6, lower bound: -587.7560896, upper bound: 587.7560896

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -326.2778931, 260.1331482, -327.0875244, 260.7777405, -587.0556030, 587.2205811
1: -275.2944031, 230.4184723, -275.9752808, 230.9897614, -506.2841492, 506.3937073
2: -360.5058594, 234.3850555, -361.3987427, 234.9588470, -595.4647217, 595.7837524
3: -381.5110474, 201.5101776, -382.4662781, 202.0105438, -583.5216064, 583.9763184
4: -351.5065308, 267.9811096, -352.3798218, 268.6433411, -620.1499023, 620.3609009
5: -314.1329651, 244.1361084, -314.9150391, 244.7410431, -558.8740234, 559.0509644
6: -300.5191040, 289.3298340, -301.2674561, 290.0480957, -590.5671997, 590.5972290
7: -327.7738953, 274.7595520, -328.5839539, 275.4401550, -603.2139282, 603.3435059
8: -395.7490540, 271.5202942, -396.7255249, 272.1857910, -667.9348145, 668.2458496
9: -298.1673584, 293.6198120, -298.9044800, 294.3448181, -592.5122070, 592.5242920

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7670699, upper bound: 587.7553521
time: 11.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7730149, upper bound: 587.7624851
time: 12.88 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -364.6564636, 290.5466614, -326.5800781, 260.3737488, -625.0300903, 617.1267090
1: -307.5503540, 257.2839661, -275.5485535, 230.6318512, -538.1821899, 532.8325195
2: -402.7117310, 261.8395386, -360.8393250, 234.5984955, -637.3102417, 622.6788330
3: -425.8497009, 224.8515472, -381.8683777, 201.6973724, -627.5470581, 606.7198486
4: -392.5126648, 299.4674683, -351.8327026, 268.2282715, -660.7409668, 651.3001709
5: -350.9626465, 272.7847900, -314.4248962, 244.3614502, -595.3240967, 587.2097168
6: -335.5522461, 323.1262207, -300.7983398, 289.5984192, -625.1506348, 623.9244385
7: -366.3593750, 306.9647522, -328.0757446, 275.0134277, -641.3727417, 635.0403442
8: -441.9480591, 303.3123169, -396.1138000, 271.7692261, -713.7172241, 699.4261475
9: -333.1286316, 327.7955017, -298.4423218, 293.8906860, -627.0192871, 626.2377319

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7450612, upper bound: 587.7439776
time: 10.15 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7433790, upper bound: 587.7433790
time: 9.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.23 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.23
Output dim: 6, lower bound: -587.7670699, upper bound: 587.7553521
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.23
Output dim: 6, lower bound: -587.7730149, upper bound: 587.7624851
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 23.23
Output dim: 6, lower bound: -587.7450612, upper bound: 587.7439776
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 23.23
Output dim: 6, lower bound: -587.7433790, upper bound: 587.7433790

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -325.4362488, 259.4671021, -328.4160767, 261.8209229, -587.2572021, 587.8830566
1: -274.5887756, 229.8279114, -277.0532227, 231.9159088, -506.5046387, 506.8811340
2: -359.5834961, 233.7908630, -362.8672485, 235.9176331, -595.5010986, 596.6580200
3: -380.5257263, 200.9965668, -383.9756165, 202.8362122, -583.3618164, 584.9721680
4: -350.6030884, 267.2940979, -353.7885742, 269.7388000, -620.3418579, 621.0826416
5: -313.3226929, 243.5102844, -316.1574402, 245.7337952, -559.0564575, 559.6676025
6: -299.7460327, 288.5863953, -302.4639282, 291.1901550, -590.9361572, 591.0502930
7: -326.9346008, 274.0557861, -329.9324341, 276.5691223, -603.5037231, 603.9882202
8: -394.7368164, 270.8327332, -398.3032837, 273.2629089, -667.9997559, 669.1359863
9: -297.4024353, 292.8694153, -300.1128235, 295.5223999, -592.9247437, 592.9822388

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7178390, upper bound: 587.7173845
time: 11.66 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7540785, upper bound: 587.7471360
time: 9.74 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7670699, upper bound: 587.7553521
time: 11.39 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -326.2778931, 260.1331482, -325.7579346, 259.7256775, -586.0035400, 585.8909912
1: -275.2944031, 230.4184723, -274.8578796, 230.0525360, -505.3469238, 505.2763062
2: -360.5058594, 234.3850555, -359.9388733, 234.0185394, -594.5244141, 594.3239136
3: -381.5110474, 201.5101776, -380.9057007, 201.1962585, -582.7072754, 582.4157715
4: -351.5065308, 267.9811096, -350.9523010, 267.5549011, -619.0614014, 618.9333496
5: -314.1329651, 244.1361084, -313.6343689, 243.7531128, -557.8861084, 557.7705078
6: -300.5191040, 289.3298340, -300.0429688, 288.8718872, -589.3909912, 589.3728027
7: -327.7738953, 274.7595520, -327.2574463, 274.3264465, -602.1002197, 602.0169067
8: -395.7490540, 271.5202942, -395.1242065, 271.0961304, -666.8451538, 666.6444702
9: -298.1673584, 293.6198120, -297.6944580, 293.1573486, -591.3247070, 591.3141479

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7353899, upper bound: 587.7334697
time: 12.51 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7638502, upper bound: 587.7568243
time: 10.94 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7730149, upper bound: 587.7624851
time: 11.27 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -348.6863098, 277.8634338, -326.5800781, 260.3737488, -609.0600586, 604.4434814
1: -294.1145630, 246.0266724, -275.5485535, 230.6318512, -524.7463989, 521.5751953
2: -385.1567078, 250.4280548, -360.8393250, 234.5984955, -619.7551270, 611.2673950
3: -407.1756897, 215.0398865, -381.8683777, 201.6973724, -608.8730469, 596.9082642
4: -375.4347229, 286.3746338, -351.8327026, 268.2282715, -643.6629639, 638.2073364
5: -335.5892944, 260.8315735, -314.4248962, 244.3614502, -579.9507446, 575.2564697
6: -320.9473267, 309.0445251, -300.7983398, 289.5984192, -610.5457764, 609.8427734
7: -350.3210449, 293.5184326, -328.0757446, 275.0134277, -625.3344116, 621.5941162
8: -422.7423706, 290.1181335, -396.1138000, 271.7692261, -694.5115967, 686.2319336
9: -318.5613708, 313.4904785, -298.4423218, 293.8906860, -612.4520264, 611.9328003

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7374608, upper bound: 587.7383897
time: 10.75 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7450612, upper bound: 587.7439776
time: 9.57 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -347.5926819, 276.9804688, -323.4600220, 257.8970947, -605.4897461, 600.4404907
1: -293.1323242, 245.2357635, -272.9245911, 228.4334412, -521.5656128, 518.1603394
2: -383.9054871, 249.6025238, -357.4074097, 232.3709106, -616.2763672, 607.0099487
3: -405.8835449, 214.3443909, -378.2156982, 199.7805176, -605.6640625, 592.5600586
4: -374.2347412, 285.4048767, -348.4879456, 265.6661377, -639.9008789, 633.8928223
5: -334.5148926, 259.9424438, -311.4223633, 242.0236053, -576.5383911, 571.3647461
6: -319.9625549, 308.0632629, -297.9429626, 286.8459473, -606.8084106, 606.0062256
7: -349.1604309, 292.5524292, -324.9371948, 272.3855896, -621.5458374, 617.4896240
8: -421.3930664, 289.1515198, -392.3627930, 269.1926575, -690.5856323, 681.5142822
9: -317.5015564, 312.4538269, -295.5892029, 291.0930786, -608.5946045, 608.0429077

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7343564, upper bound: 587.7373421
time: 10.15 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7433790, upper bound: 587.7433790
time: 9.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.28 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 6, lower bound: -587.7540785, upper bound: 587.7471360
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 6, lower bound: -587.7670699, upper bound: 587.7553521
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 6, lower bound: -587.7638502, upper bound: 587.7568243
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 6, lower bound: -587.7730149, upper bound: 587.7624851
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 6, lower bound: -587.7374608, upper bound: 587.7383897
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 6, lower bound: -587.7450612, upper bound: 587.7439776
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 6, lower bound: -587.7343564, upper bound: 587.7373421
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 6, lower bound: -587.7433790, upper bound: 587.7433790

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -328.0563965, 261.5494080, -327.4605103, 261.0661926, -589.1225586, 589.0098877
1: -276.8175659, 231.6995850, -276.2534790, 231.2470703, -508.0646057, 507.9530334
2: -362.4884033, 235.6568604, -361.8185730, 235.2428284, -597.7312012, 597.4754639
3: -383.5748291, 202.6446381, -382.8576660, 202.2521820, -585.8270264, 585.5022583
4: -353.3564758, 269.4372559, -352.7618103, 268.9587708, -622.3151245, 622.1990967
5: -315.8496399, 245.4736023, -315.2393494, 245.0236053, -560.8732300, 560.7127686
6: -302.0640564, 290.9021606, -301.5807800, 290.3465271, -592.4105225, 592.4829102
7: -329.6175232, 276.2451477, -328.9790649, 275.7685242, -605.3860474, 605.2241821
8: -397.9024658, 272.9775085, -397.1563721, 272.4852905, -670.3877563, 670.1337280
9: -299.8292847, 295.2242737, -299.2463379, 294.6708984, -594.5001831, 594.4705200

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7492557, upper bound: 587.7424465
time: 12.27 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7492557, upper bound: 587.7471360
time: 11.15 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -324.1537476, 258.4543152, -328.4160767, 261.8209229, -585.9746704, 586.8703003
1: -273.5152588, 228.9306946, -277.0532227, 231.9159088, -505.4311218, 505.9838867
2: -358.1745911, 232.8847961, -362.8672485, 235.9176331, -594.0922241, 595.7519531
3: -379.0225525, 200.2123566, -383.9756165, 202.8362122, -581.8587036, 584.1879883
4: -349.2232666, 266.2450867, -353.7885742, 269.7388000, -618.9620361, 620.0336914
5: -312.0911255, 242.5567474, -316.1574402, 245.7337952, -557.8249512, 558.7141113
6: -298.5628052, 287.4544067, -302.4639282, 291.1901550, -589.7528687, 589.9183350
7: -325.6531982, 272.9809265, -329.9324341, 276.5691223, -602.2221680, 602.9133301
8: -393.1967468, 269.7857361, -398.3032837, 273.2629089, -666.4596558, 668.0889893
9: -296.2375488, 291.7265015, -300.1128235, 295.5223999, -591.7598877, 591.8392944

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7178390, upper bound: 587.7173845
time: 13.92 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7523170, upper bound: 587.7464928
time: 12.69 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7658676, upper bound: 587.7541589
time: 12.28 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7649038, upper bound: 587.7524932
time: 11.88 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -328.9076538, 262.2231750, -324.8110657, 258.9777222, -587.8853760, 587.0342407
1: -277.5315552, 232.2968750, -274.0657043, 229.3898315, -506.9213867, 506.3625488
2: -363.4210815, 236.2577667, -358.8996582, 233.3497314, -596.7708130, 595.1572876
3: -384.5709229, 203.1641541, -379.7977600, 200.6175842, -585.1884766, 582.9618530
4: -354.2696838, 270.1321106, -349.9345703, 266.7820129, -621.0516968, 620.0665894
5: -316.6692200, 246.1067657, -312.7245789, 243.0496063, -559.7188110, 558.8313599
6: -302.8458252, 291.6541138, -299.1676636, 288.0360718, -590.8818359, 590.8217773
7: -330.4662170, 276.9570618, -326.3128052, 273.5331726, -603.9993286, 603.2697144
8: -398.9261169, 273.6727600, -393.9876709, 270.3253479, -669.2513428, 667.6603394
9: -300.6029663, 295.9834290, -296.8358765, 292.3138123, -592.9167480, 592.8193359

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7492557, upper bound: 587.7490079
time: 12.36 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7492557, upper bound: 587.7568243
time: 11.49 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -324.9935608, 259.1190796, -325.7579346, 259.7256775, -584.7192383, 584.8768921
1: -274.2194824, 229.5200500, -274.8578796, 230.0525360, -504.2720337, 504.3779297
2: -359.0950317, 233.4778442, -359.9388733, 234.0185394, -593.1135864, 593.4166260
3: -380.0057983, 200.7248993, -380.9057007, 201.1962585, -581.2019043, 581.6304932
4: -350.1248169, 266.9307251, -350.9523010, 267.5549011, -617.6796875, 617.8830566
5: -312.8999023, 243.1813354, -313.6343689, 243.7531128, -556.6530151, 556.8156738
6: -299.3342590, 288.1963196, -300.0429688, 288.8718872, -588.2061768, 588.2392578
7: -326.4907227, 273.6833191, -327.2574463, 274.3264465, -600.8170776, 600.9407349
8: -394.2069702, 270.4719543, -395.1242065, 271.0961304, -665.3031006, 665.5960693
9: -297.0009460, 292.4753723, -297.6944580, 293.1573486, -590.1582642, 590.1697998

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7353899, upper bound: 587.7334697
time: 12.02 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7606026, upper bound: 587.7550564
time: 11.18 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7453652, upper bound: 587.7293202
time: 11.81 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -350.6140137, 279.3851929, -325.7388916, 259.7080078, -610.3220215, 605.1240845
1: -295.7089844, 247.3756561, -274.8432312, 230.0415802, -525.7505493, 522.2188721
2: -387.2984619, 251.8228760, -359.9174194, 234.0046082, -621.3031006, 611.7402954
3: -409.3873291, 216.2362213, -380.8834839, 201.1839905, -610.5712891, 597.1196899
4: -377.5053711, 287.9640503, -350.9297180, 267.5415039, -645.0468750, 638.8937988
5: -337.4137573, 262.2736816, -313.6149597, 243.7359467, -581.1496582, 575.8886719
6: -322.6947021, 310.7276001, -300.0256042, 288.8553467, -611.5500488, 610.7531738
7: -352.2933044, 295.1518555, -327.2369385, 274.3099976, -626.6032104, 622.3887939
8: -425.0582275, 291.6922302, -395.1020203, 271.0819702, -696.1401367, 686.7942505
9: -320.3283081, 315.2143250, -297.6777344, 293.1406860, -613.4688721, 612.8920898

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A1_A1_A1

### Relational analysis result of IS_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7120047, upper bound: 587.7061839
time: 10.92 seconds

## Relational analysis of IS_A2_A1_A1_A2

### Relational analysis result of IS_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7016592, upper bound: 587.7001875
time: 10.25 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -347.3351746, 276.7944031, -326.5800781, 260.3737488, -607.7088623, 603.3745117
1: -292.9791870, 245.0741882, -275.5485535, 230.6318512, -523.6110229, 520.6226807
2: -383.6734619, 249.4721832, -360.8393250, 234.5984955, -618.2719727, 610.3115234
3: -405.5899048, 214.2125854, -381.8683777, 201.6973724, -607.2872925, 596.0808716
4: -373.9838562, 285.2687073, -351.8327026, 268.2282715, -642.2119751, 637.1014404
5: -334.2883301, 259.8275452, -314.4248962, 244.3614502, -578.6495972, 574.2524414
6: -319.7030334, 307.8492737, -300.7983398, 289.5984192, -609.3013306, 608.6475220
7: -348.9721985, 292.3866577, -328.0757446, 275.0134277, -623.9855957, 620.4623413
8: -421.1153870, 289.0107117, -396.1138000, 271.7692261, -692.8845825, 685.1244507
9: -317.3314819, 312.2834778, -298.4423218, 293.8906860, -611.2221680, 610.7258301

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A1_A2_A1

### Relational analysis result of IS_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7231801, upper bound: 587.7143122
time: 9.98 seconds

## Relational analysis of IS_A2_A1_A2_A2

### Relational analysis result of IS_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7170322, upper bound: 587.7115553
time: 9.13 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -350.0509033, 278.9243164, -322.6189270, 257.2314148, -607.2823486, 601.5430908
1: -295.1616516, 246.9554596, -272.2193604, 227.8432007, -523.0048218, 519.1748047
2: -386.6267395, 251.3791809, -356.4856262, 231.7770386, -618.4038086, 607.8648071
3: -408.7084045, 215.8534241, -377.2308350, 199.2672272, -607.9755249, 593.0842285
4: -376.8660889, 287.4293213, -347.5850525, 264.9795227, -641.8455811, 635.0143433
5: -336.8448792, 261.7747498, -310.6123657, 241.3981628, -578.2429810, 572.3870850
6: -322.1960449, 310.2037354, -297.1703186, 286.1029053, -608.2989502, 607.3739624
7: -351.6604004, 294.6349487, -324.0985107, 271.6821899, -623.3424683, 618.7334595
8: -424.3395996, 291.1646729, -391.3511353, 268.5055237, -692.8450317, 682.5158081
9: -319.7474670, 314.6442261, -294.8246765, 290.3431702, -610.0906372, 609.4688110

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7092514, upper bound: 587.7048167
time: 11.47 seconds

## Relational analysis of IS_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.6709276, upper bound: 587.6790840
time: 11.65 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -346.2069397, 275.8842163, -323.4600220, 257.8970947, -604.1040039, 599.3442383
1: -291.9680786, 244.2590027, -272.9245911, 228.4334412, -520.4014282, 517.1835938
2: -382.3844299, 248.6221924, -357.4074097, 232.3709106, -614.7553711, 606.0296021
3: -404.2572937, 213.4959564, -378.2156982, 199.7805176, -604.0378418, 591.7116089
4: -372.7469177, 284.2707825, -348.4879456, 265.6661377, -638.4130249, 632.7586060
5: -333.1808167, 258.9126282, -311.4223633, 242.0236053, -575.2042847, 570.3349609
6: -318.6860657, 306.8376160, -297.9429626, 286.8459473, -605.5319214, 604.7805176
7: -347.7769165, 291.3917236, -324.9371948, 272.3855896, -620.1624146, 616.3289185
8: -419.7247620, 288.0157166, -392.3627930, 269.1926575, -688.9172974, 680.3784180
9: -316.2400818, 311.2158813, -295.5892029, 291.0930786, -607.3331299, 606.8049316

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7216943, upper bound: 587.7137813
time: 10.34 seconds

## Relational analysis of IS_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7092324, upper bound: 587.7092324
time: 10.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.75 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7492557, upper bound: 587.7424465
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7492557, upper bound: 587.7471360
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7658676, upper bound: 587.7541589
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7649038, upper bound: 587.7524932
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7492557, upper bound: 587.7490079
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7492557, upper bound: 587.7568243
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7606026, upper bound: 587.7550564
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7453652, upper bound: 587.7293202
IS_A2_A1_A1_A1, status: Status.VERIFIED, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7120047, upper bound: 587.7061839
IS_A2_A1_A1_A2, status: Status.VERIFIED, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7016592, upper bound: 587.7001875
IS_A2_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7231801, upper bound: 587.7143122
IS_A2_A1_A2_A2, status: Status.VERIFIED, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7170322, upper bound: 587.7115553
IS_A2_A2_A1_A1, status: Status.VERIFIED, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7092514, upper bound: 587.7048167
IS_A2_A2_A1_A2, status: Status.VERIFIED, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.6709276, upper bound: 587.6790840
IS_A2_A2_A2_A1, status: Status.VERIFIED, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7216943, upper bound: 587.7137813
IS_A2_A2_A2_A2, status: Status.VERIFIED, split count: 4, time: 23.75
Output dim: 6, lower bound: -587.7092324, upper bound: 587.7092324

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -329.4055481, 262.6135559, -327.4605103, 261.0661926, -590.4717407, 590.0740967
1: -277.9250488, 232.6453857, -276.2534790, 231.2470703, -509.1720886, 508.8988342
2: -363.9869080, 236.6303558, -361.8185730, 235.2428284, -599.2296753, 598.4489136
3: -385.1343079, 203.4937134, -382.8576660, 202.2521820, -587.3864746, 586.3513794
4: -354.8126831, 270.5572205, -352.7618103, 268.9587708, -623.7714233, 623.3190308
5: -317.1211243, 246.4817810, -315.2393494, 245.0236053, -562.1447144, 561.7210083
6: -303.2945862, 292.0760803, -301.5807800, 290.3465271, -593.6411133, 593.6568604
7: -330.9885559, 277.3947754, -328.9790649, 275.7685242, -606.7570801, 606.3737793
8: -399.5167847, 274.0877686, -397.1563721, 272.4852905, -672.0020752, 671.2441406
9: -301.0615540, 296.4337463, -299.2463379, 294.6708984, -595.7324219, 595.6799927

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7359668, upper bound: 587.7277226
time: 10.14 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 218

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 218

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7491759, upper bound: 587.7422115
time: 10.91 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7426177, upper bound: 587.7358404
time: 11.55 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -327.5758362, 261.1693420, -327.4605103, 261.0661926, -588.6420288, 588.6298828
1: -276.4121704, 231.3580475, -276.2534790, 231.2470703, -507.6592102, 507.6115112
2: -361.9589539, 235.3158722, -361.8185730, 235.2428284, -597.2017822, 597.1344604
3: -383.0079651, 202.3483887, -382.8576660, 202.2521820, -585.2601318, 585.2060547
4: -352.8402405, 269.0419922, -352.7618103, 268.9587708, -621.7990112, 621.8038330
5: -315.3864136, 245.1169281, -315.2393494, 245.0236053, -560.4100342, 560.3562012
6: -301.6191711, 290.4758606, -301.5807800, 290.3465271, -591.9656982, 592.0566406
7: -329.1374512, 275.8413696, -328.9790649, 275.7685242, -604.9060059, 604.8204346
8: -397.3222351, 272.5813599, -397.1563721, 272.4852905, -669.8074341, 669.7376709
9: -299.3909302, 294.7937012, -299.2463379, 294.6708984, -594.0618286, 594.0400391

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=591.3155517578125
rel_dist={6: [-587.7907620297522, 587.7907620249766]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7599586, upper bound: 587.7658051
time: 12.62 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7560364, upper bound: 587.7560364
time: 11.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 24.09 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 24.09
Output dim: 6, lower bound: -587.7599586, upper bound: 587.7658051
IS_B2, status: Status.UNKNOWN, split count: 1, time: 24.09
Output dim: 6, lower bound: -587.7560364, upper bound: 587.7560364

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -327.0875244, 260.7777405, -326.2778931, 260.1331482, -587.2205811, 587.0556030
1: -275.9752808, 230.9897614, -275.2944031, 230.4184723, -506.3937073, 506.2841492
2: -361.3987427, 234.9588470, -360.5058594, 234.3850555, -595.7837524, 595.4647217
3: -382.4662781, 202.0105438, -381.5110474, 201.5101776, -583.9763184, 583.5216064
4: -352.3798218, 268.6433411, -351.5065308, 267.9811096, -620.3609009, 620.1499023
5: -314.9150391, 244.7410431, -314.1329651, 244.1361084, -559.0509644, 558.8740234
6: -301.2674561, 290.0480957, -300.5191040, 289.3298340, -590.5972290, 590.5671997
7: -328.5839539, 275.4401550, -327.7738953, 274.7595520, -603.3435059, 603.2139282
8: -396.7255249, 272.1857910, -395.7490540, 271.5202942, -668.2458496, 667.9348145
9: -298.9044800, 294.3448181, -298.1673584, 293.6198120, -592.5242920, 592.5122070

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 203

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.6919565, upper bound: 587.6923018
time: 13.44 seconds

## Relational analysis of IS_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7520782, upper bound: 587.7577971
time: 12.79 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7599586, upper bound: 587.7658051
time: 11.90 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -325.8082581, 259.7592163, -364.6564636, 290.5466614, -616.3548584, 624.4154663
1: -274.8994141, 230.0874634, -307.5503540, 257.2839661, -532.1833496, 537.6378174
2: -359.9884033, 234.0503998, -402.7117310, 261.8395386, -621.8279419, 636.7621460
3: -380.9587708, 201.2210388, -425.8497009, 224.8515472, -605.8103027, 627.0707397
4: -351.0005188, 267.5968628, -392.5126648, 299.4674683, -650.4680176, 660.1094971
5: -313.6793823, 243.7840576, -350.9626465, 272.7847900, -586.4641724, 594.7467041
6: -300.0847168, 288.9144897, -335.5522461, 323.1262207, -623.2108154, 624.4667358
7: -327.3025818, 274.3643494, -366.3593750, 306.9647522, -634.2673340, 640.7236328
8: -395.1831970, 271.1354980, -441.9480591, 303.3123169, -698.4954834, 713.0834961
9: -297.7392883, 293.1998596, -333.1286316, 327.7955017, -625.5346680, 626.3284912

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7436731, upper bound: 587.7443713
time: 10.72 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7433256, upper bound: 587.7433256
time: 10.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.30 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 24.30
Output dim: 6, lower bound: -587.7520782, upper bound: 587.7577971
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 24.30
Output dim: 6, lower bound: -587.7599586, upper bound: 587.7658051
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 24.30
Output dim: 6, lower bound: -587.7436731, upper bound: 587.7443713
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 24.30
Output dim: 6, lower bound: -587.7433256, upper bound: 587.7433256

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -328.4160767, 261.8209229, -324.0012512, 258.3312683, -586.7473145, 585.8221436
1: -277.0532227, 231.9159088, -273.3854980, 228.8208313, -505.8740540, 505.3013611
2: -362.8672485, 235.9176331, -358.0105896, 232.7776947, -595.6448364, 593.9282227
3: -383.9756165, 202.8362122, -378.8453979, 200.1207123, -584.0962524, 581.6815186
4: -353.7885742, 269.7388000, -349.0625305, 266.1224670, -619.9110107, 618.8012085
5: -316.1574402, 245.7337952, -311.9408875, 242.4431458, -558.6005249, 557.6745605
6: -302.4639282, 291.1901550, -298.4277954, 287.3186646, -589.7825928, 589.6179199
7: -329.9324341, 276.5691223, -325.5033875, 272.8556519, -602.7880859, 602.0724487
8: -398.3032837, 273.2629089, -393.0104980, 269.6602478, -667.9635010, 666.2734375
9: -300.1128235, 295.5223999, -296.0979309, 291.5898132, -591.7026367, 591.6202393

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7571653
time: 11.91 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7577971
time: 12.43 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -325.7579346, 259.7256775, -326.1943665, 260.0670776, -585.8249512, 585.9199829
1: -274.8578796, 230.0525360, -275.2241516, 230.3595886, -505.2174683, 505.2766724
2: -359.9388733, 234.0185394, -360.4141541, 234.3260040, -594.2648926, 594.4326782
3: -380.9057007, 201.1962585, -381.4130554, 201.4590607, -582.3646240, 582.6092529
4: -350.9523010, 267.5549011, -351.4168396, 267.9127197, -618.8649902, 618.9717407
5: -313.6343689, 243.7531128, -314.0525818, 244.0740356, -557.7083740, 557.8056030
6: -300.0429688, 288.8718872, -300.4422302, 289.2559204, -589.2988892, 589.3140869
7: -327.2574463, 274.3264465, -327.6905212, 274.6896362, -601.9470825, 602.0169678
8: -395.1242065, 271.0961304, -395.6484985, 271.4518433, -666.5760498, 666.7445068
9: -297.6944580, 293.1573486, -298.0913086, 293.5451965, -591.2396240, 591.2486572

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7602907
time: 11.48 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7658051
time: 14.07 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -320.5311584, 255.5693665, -348.6863098, 277.8634338, -598.3945923, 604.2556152
1: -270.4643860, 226.3687592, -294.1145630, 246.0266724, -516.4909668, 520.4832764
2: -354.1912231, 230.2827454, -385.1567078, 250.4280548, -604.6192017, 615.4393311
3: -374.7903748, 197.9810028, -407.1756897, 215.0398865, -589.8302612, 605.1566772
4: -345.3620911, 263.2721558, -375.4347229, 286.3746338, -631.7366943, 638.7069092
5: -308.5992737, 239.8352356, -335.5892944, 260.8315735, -569.4307861, 575.4245605
6: -295.2605286, 284.2639465, -320.9473267, 309.0445251, -604.3050537, 605.2113037
7: -322.0068359, 269.9227295, -350.3210449, 293.5184326, -615.5252686, 620.2436523
8: -388.8420715, 266.7778931, -422.7423706, 290.1181335, -678.9602051, 689.5202637
9: -292.9266357, 288.4766541, -318.5613708, 313.4904785, -606.4171143, 607.0380249

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7119823, upper bound: 587.7174526
time: 10.94 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7104477, upper bound: 587.7139324
time: 11.58 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -315.6685791, 251.7125397, -347.5926819, 276.9804688, -592.6490479, 599.3051758
1: -266.3717041, 222.9446411, -293.1323242, 245.2357635, -511.6074829, 516.0768433
2: -348.8347778, 226.8135529, -383.9054871, 249.6025238, -598.4373169, 610.7190552
3: -369.0840759, 194.9904633, -405.8835449, 214.3443909, -583.4284058, 600.8739624
4: -340.1319885, 259.2688293, -374.2347412, 285.4048767, -625.5368042, 633.5035400
5: -303.9220886, 236.1876373, -334.5148926, 259.9424438, -563.8645020, 570.7023926
6: -290.8057251, 279.9674377, -319.9625549, 308.0632629, -598.8690186, 599.9299316
7: -317.1048584, 265.8222656, -349.1604309, 292.5524292, -609.6572876, 614.9826050
8: -382.9946594, 262.7669067, -421.3930664, 289.1515198, -672.1461792, 684.1599731
9: -288.4681396, 284.1082153, -317.5015564, 312.4538269, -600.9219971, 601.6097412

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7117154, upper bound: 587.7165831
time: 11.47 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7091931, upper bound: 587.7091931
time: 10.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.10 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 25.10
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7571653
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 25.10
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7577971
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.10
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7602907
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.10
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7658051
IS_B2_B1_B1, status: Status.VERIFIED, split count: 3, time: 25.10
Output dim: 6, lower bound: -587.7119823, upper bound: 587.7174526
IS_B2_B1_B2, status: Status.VERIFIED, split count: 3, time: 25.10
Output dim: 6, lower bound: -587.7104477, upper bound: 587.7139324
IS_B2_B2_B1, status: Status.VERIFIED, split count: 3, time: 25.10
Output dim: 6, lower bound: -587.7117154, upper bound: 587.7165831
IS_B2_B2_B2, status: Status.VERIFIED, split count: 3, time: 25.10
Output dim: 6, lower bound: -587.7091931, upper bound: 587.7091931

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -328.4160767, 261.8209229, -327.6079102, 261.1775208, -589.5935669, 589.4287109
1: -277.0532227, 231.9159088, -276.3737793, 231.3455963, -508.3988037, 508.2896423
2: -362.8672485, 235.9176331, -361.9759521, 235.3447418, -598.2119141, 597.8935547
3: -383.9756165, 202.8362122, -383.0221558, 202.3367310, -586.3123779, 585.8582764
4: -353.7885742, 269.7388000, -352.9167480, 269.0776367, -622.8662109, 622.6554565
5: -316.1574402, 245.7337952, -315.3769836, 245.1299591, -561.2874146, 561.1106567
6: -302.4639282, 291.1901550, -301.7168579, 290.4729614, -592.9368286, 592.9069214
7: -329.9324341, 276.5691223, -329.1233826, 275.8897400, -605.8221436, 605.6924438
8: -398.3032837, 273.2629089, -397.3283081, 272.5983887, -670.9016724, 670.5911865
9: -300.1128235, 295.5223999, -299.3767395, 294.7984314, -594.9112549, 594.8991089

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 211

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7571653
time: 11.97 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7571653
time: 10.88 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -328.4160767, 261.8209229, -324.9497070, 259.0822144, -587.4981689, 586.7706299
1: -277.0532227, 231.9159088, -274.1781921, 229.4822083, -506.5354309, 506.0940552
2: -362.8672485, 235.9176331, -359.0476074, 233.4457855, -596.3129883, 594.9652100
3: -383.9756165, 202.8362122, -379.9520874, 200.6967468, -584.6723633, 582.7882080
4: -353.7885742, 269.7388000, -350.0805664, 266.8938293, -620.6823730, 619.8193359
5: -316.1574402, 245.7337952, -312.8537598, 243.1492157, -559.3065796, 558.5874634
6: -302.4639282, 291.1901550, -299.2959290, 288.1547241, -590.6186523, 590.4860229
7: -329.9324341, 276.5691223, -326.4486389, 273.6470337, -603.5794678, 603.0177002
8: -398.3032837, 273.2629089, -394.1493530, 270.4316711, -668.7349854, 667.4122314
9: -300.1128235, 295.5223999, -296.9585571, 292.4334717, -592.5462646, 592.4808960

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7577971
time: 11.61 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7577971
time: 13.84 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -325.7579346, 259.7256775, -327.6079102, 261.1775208, -586.9354248, 587.3333740
1: -274.8578796, 230.0525360, -276.3737793, 231.3455963, -506.2034912, 506.4263306
2: -359.9388733, 234.0185394, -361.9759521, 235.3447418, -595.2836304, 595.9945068
3: -380.9057007, 201.1962585, -383.0221558, 202.3367310, -583.2424316, 584.2183228
4: -350.9523010, 267.5549011, -352.9167480, 269.0776367, -620.0299072, 620.4716187
5: -313.6343689, 243.7531128, -315.3769836, 245.1299591, -558.7643433, 559.1300049
6: -300.0429688, 288.8718872, -301.7168579, 290.4729614, -590.5159302, 590.5886841
7: -327.2574463, 274.3264465, -329.1233826, 275.8897400, -603.1470947, 603.4498291
8: -395.1242065, 271.0961304, -397.3283081, 272.5983887, -667.7225952, 668.4244385
9: -297.6944580, 293.1573486, -299.3767395, 294.7984314, -592.4929199, 592.5340576

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7602907
time: 12.48 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7602907
time: 12.87 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -325.7579346, 259.7256775, -324.9497070, 259.0822144, -584.8400269, 584.6752930
1: -274.8578796, 230.0525360, -274.1781921, 229.4822083, -504.3400879, 504.2307129
2: -359.9388733, 234.0185394, -359.0476074, 233.4457855, -593.3846436, 593.0661621
3: -380.9057007, 201.1962585, -379.9520874, 200.6967468, -581.6022949, 581.1482544
4: -350.9523010, 267.5549011, -350.0805664, 266.8938293, -617.8461304, 617.6354980
5: -313.6343689, 243.7531128, -312.8537598, 243.1492157, -556.7835693, 556.6068115
6: -300.0429688, 288.8718872, -299.2959290, 288.1547241, -588.1976929, 588.1677856
7: -327.2574463, 274.3264465, -326.4486389, 273.6470337, -600.9044800, 600.7750854
8: -395.1242065, 271.0961304, -394.1493530, 270.4316711, -665.5559082, 665.2454834
9: -297.6944580, 293.1573486, -296.9585571, 292.4334717, -590.1278687, 590.1159058

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7658051
time: 11.09 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7658051
time: 12.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 34.92 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 34.92
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7571653
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 34.92
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7571653
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.92
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7577971
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.92
Output dim: 6, lower bound: -587.7518554, upper bound: 587.7577971
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 34.92
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7602907
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 34.92
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7602907
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.92
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7658051
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.92
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7658051

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -327.6079102, 261.1775208, -327.6079102, 261.1775208, -588.7852783, 588.7852783
1: -276.3737793, 231.3455963, -276.3737793, 231.3455963, -507.7193604, 507.7193604
2: -361.9759521, 235.3447418, -361.9759521, 235.3447418, -597.3206787, 597.3206787
3: -383.0221558, 202.3367310, -383.0221558, 202.3367310, -585.3588867, 585.3588867
4: -352.9167480, 269.0776367, -352.9167480, 269.0776367, -621.9943848, 621.9943848
5: -315.3769836, 245.1299591, -315.3769836, 245.1299591, -560.5068970, 560.5068970
6: -301.7168579, 290.4729614, -301.7168579, 290.4729614, -592.1898193, 592.1898193
7: -329.1233826, 275.8897400, -329.1233826, 275.8897400, -605.0131226, 605.0131226
8: -397.3283081, 272.5983887, -397.3283081, 272.5983887, -669.9266968, 669.9266968
9: -299.3767395, 294.7984314, -299.3767395, 294.7984314, -594.1751709, 594.1751709

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 218

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 218

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7242203, upper bound: 587.7271655
time: 12.52 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7483459, upper bound: 587.7545667
time: 13.12 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7483219, upper bound: 587.7545200
time: 14.07 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -366.6179810, 292.0991821, -327.6079102, 261.1775208, -627.7955322, 619.7069702
1: -309.1633301, 258.6566772, -276.3737793, 231.3455963, -540.5088501, 535.0303955
2: -404.8877258, 263.2563171, -361.9759521, 235.3447418, -640.2324829, 625.2322998
3: -428.0937805, 226.0554657, -383.0221558, 202.3367310, -630.4305420, 609.0776367
4: -394.6141663, 301.0840454, -352.9167480, 269.0776367, -663.6917725, 654.0006714
5: -352.8173828, 274.2520752, -315.3769836, 245.1299591, -597.9473267, 589.6289673
6: -337.3248291, 324.8309937, -301.7168579, 290.4729614, -627.7977905, 626.5477905
7: -368.3590698, 308.6270447, -329.1233826, 275.8897400, -644.2487793, 637.7504272
8: -444.3009949, 304.9144592, -397.3283081, 272.5983887, -716.8993530, 702.2427979
9: -334.9216309, 329.5488586, -299.3767395, 294.7984314, -629.7200928, 628.9255371

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7394490, upper bound: 587.7439109
time: 11.83 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7376877, upper bound: 587.7435090
time: 11.75 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -327.6079102, 261.1775208, -324.9497070, 259.0822144, -586.6898804, 586.1271973
1: -276.3737793, 231.3455963, -274.1781921, 229.4822083, -505.8559875, 505.5238037
2: -361.9759521, 235.3447418, -359.0476074, 233.4457855, -595.4216919, 594.3923340
3: -383.0221558, 202.3367310, -379.9520874, 200.6967468, -583.7188110, 582.2888184
4: -352.9167480, 269.0776367, -350.0805664, 266.8938293, -619.8104858, 619.1582031
5: -315.3769836, 245.1299591, -312.8537598, 243.1492157, -558.5261230, 557.9837036
6: -301.7168579, 290.4729614, -299.2959290, 288.1547241, -589.8715820, 589.7689209
7: -329.1233826, 275.8897400, -326.4486389, 273.6470337, -602.7703857, 602.3383789
8: -397.3283081, 272.5983887, -394.1493530, 270.4316711, -667.7600098, 666.7477417
9: -299.3767395, 294.7984314, -296.9585571, 292.4334717, -591.8101807, 591.7569580

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7280326, upper bound: 587.7300436
time: 13.22 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 218

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 218

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7499445, upper bound: 587.7558409
time: 13.06 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7484976, upper bound: 587.7549356
time: 13.56 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -366.6179810, 292.0991821, -324.9497070, 259.0822144, -625.7001953, 617.0488281
1: -309.1633301, 258.6566772, -274.1781921, 229.4822083, -538.6455078, 532.8348389
2: -404.8877258, 263.2563171, -359.0476074, 233.4457855, -638.3334961, 622.3039551
3: -428.0937805, 226.0554657, -379.9520874, 200.6967468, -628.7905273, 606.0075073
4: -394.6141663, 301.0840454, -350.0805664, 266.8938293, -661.5078735, 651.1646118
5: -352.8173828, 274.2520752, -312.8537598, 243.1492157, -595.9666138, 587.1057739
6: -337.3248291, 324.8309937, -299.2959290, 288.1547241, -625.4795532, 624.1268921
7: -368.3590698, 308.6270447, -326.4486389, 273.6470337, -642.0061035, 635.0756836
8: -444.3009949, 304.9144592, -394.1493530, 270.4316711, -714.7326660, 699.0638428
9: -334.9216309, 329.5488586, -296.9585571, 292.4334717, -627.3551025, 626.5073853

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7403761, upper bound: 587.7448502
time: 13.66 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7383052, upper bound: 587.7442987
time: 12.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 32.00 seconds
IS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.00
Output dim: 6, lower bound: -587.7483459, upper bound: 587.7545667
IS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.00
Output dim: 6, lower bound: -587.7483219, upper bound: 587.7545200
IS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 32.00
Output dim: 6, lower bound: -587.7394490, upper bound: 587.7439109
IS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 32.00
Output dim: 6, lower bound: -587.7376877, upper bound: 587.7435090
IS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 32.00
Output dim: 6, lower bound: -587.7499445, upper bound: 587.7558409
IS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 32.00
Output dim: 6, lower bound: -587.7484976, upper bound: 587.7549356
IS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 32.00
Output dim: 6, lower bound: -587.7403761, upper bound: 587.7448502
IS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 32.00
Output dim: 6, lower bound: -587.7383052, upper bound: 587.7442987
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 32.00
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7602907
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 32.00
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7602907
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 32.00
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7658051
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 32.00
Output dim: 6, lower bound: -587.7562927, upper bound: 587.7658051
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=591.3155517578125
rel_dist={6: [-587.7906229930563, 587.7906229976039]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7570502, upper bound: 587.7590029
time: 18.72 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7555821, upper bound: 587.7555821
time: 14.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 33.21 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 33.21
Output dim: 6, lower bound: -587.7570502, upper bound: 587.7590029
IS_B2, status: Status.UNKNOWN, split count: 1, time: 33.21
Output dim: 6, lower bound: -587.7555821, upper bound: 587.7555821

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -326.9221802, 260.6460571, -326.2778931, 260.1331482, -587.0552979, 586.9239502
1: -275.8362122, 230.8730774, -275.2944031, 230.4184723, -506.2546387, 506.1674194
2: -361.2163696, 234.8416290, -360.5058594, 234.3850555, -595.6013794, 595.3474731
3: -382.2711182, 201.9083557, -381.5110474, 201.5101776, -583.7811890, 583.4193115
4: -352.2014771, 268.5080566, -351.5065308, 267.9811096, -620.1824951, 620.0145874
5: -314.7552490, 244.6174622, -314.1329651, 244.1361084, -558.8912964, 558.7504272
6: -301.1145630, 289.9013977, -300.5191040, 289.3298340, -590.4443970, 590.4205322
7: -328.4185791, 275.3011475, -327.7738953, 274.7595520, -603.1781006, 603.0749512
8: -396.5260925, 272.0498657, -395.7490540, 271.5202942, -668.0463257, 667.7989502
9: -298.7538757, 294.1967468, -298.1673584, 293.6198120, -592.3735962, 592.3641357

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7555821, upper bound: 587.7555821
time: 13.85 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7555821, upper bound: 587.7555821
time: 12.70 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -324.2521057, 258.5212097, -364.6564636, 290.5466614, -614.7987671, 623.1776733
1: -273.5916748, 228.9904785, -307.5503540, 257.2839661, -530.8756104, 536.5408325
2: -358.2735901, 232.9457703, -402.7117310, 261.8395386, -620.1130371, 635.6574707
3: -379.1264343, 200.2622375, -425.8497009, 224.8515472, -603.9779663, 626.1119385
4: -349.3234558, 266.3244019, -392.5126648, 299.4674683, -648.7908936, 658.8370361
5: -312.1771851, 242.6205597, -350.9626465, 272.7847900, -584.9619751, 593.5831299
6: -298.6461792, 287.5368652, -335.5522461, 323.1262207, -621.7723389, 623.0890503
7: -325.7441711, 273.0563660, -366.3593750, 306.9647522, -632.7088013, 639.4156494
8: -393.3084106, 269.8598633, -441.9480591, 303.3123169, -696.6207275, 711.8078613
9: -296.3224182, 291.8082275, -333.1286316, 327.7955017, -624.1177979, 624.9368896

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7431786, upper bound: 587.7434840
time: 12.35 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7430441, upper bound: 587.7430441
time: 12.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.06 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 28.06
Output dim: 6, lower bound: -587.7555821, upper bound: 587.7555821
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 28.06
Output dim: 6, lower bound: -587.7555821, upper bound: 587.7555821
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 28.06
Output dim: 6, lower bound: -587.7431786, upper bound: 587.7434840
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 28.06
Output dim: 6, lower bound: -587.7430441, upper bound: 587.7430441

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -326.2778931, 260.1331482, -326.2778931, 260.1331482, -586.4110107, 586.4110107
1: -275.2944031, 230.4184723, -275.2944031, 230.4184723, -505.7127991, 505.7127991
2: -360.5058594, 234.3850555, -360.5058594, 234.3850555, -594.8908691, 594.8908691
3: -381.5110474, 201.5101776, -381.5110474, 201.5101776, -583.0211792, 583.0211792
4: -351.5065308, 267.9811096, -351.5065308, 267.9811096, -619.4876709, 619.4876709
5: -314.1329651, 244.1361084, -314.1329651, 244.1361084, -558.2690430, 558.2690430
6: -300.5191040, 289.3298340, -300.5191040, 289.3298340, -589.8489380, 589.8489380
7: -327.7738953, 274.7595520, -327.7738953, 274.7595520, -602.5333252, 602.5333252
8: -395.7490540, 271.5202942, -395.7490540, 271.5202942, -667.2693481, 667.2693481
9: -298.1673584, 293.6198120, -298.1673584, 293.6198120, -591.7871704, 591.7871704

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 136

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 235

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 235

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7479441, upper bound: 587.7499823
time: 15.52 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7570502, upper bound: 587.7590029
time: 18.52 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -364.6564636, 290.5466614, -326.2778931, 260.1331482, -624.7894897, 616.8245850
1: -307.5503540, 257.2839661, -275.2944031, 230.4184723, -537.9688110, 532.5783691
2: -402.7117310, 261.8395386, -360.5058594, 234.3850555, -637.0968018, 622.3453369
3: -425.8497009, 224.8515472, -381.5110474, 201.5101776, -627.3598633, 606.3626099
4: -392.5126648, 299.4674683, -351.5065308, 267.9811096, -660.4937744, 650.9739990
5: -350.9626465, 272.7847900, -314.1329651, 244.1361084, -595.0986938, 586.9177246
6: -335.5522461, 323.1262207, -300.5191040, 289.3298340, -624.8820801, 623.6452637
7: -366.3593750, 306.9647522, -327.7738953, 274.7595520, -641.1188965, 634.7385254
8: -441.9480591, 303.3123169, -395.7490540, 271.5202942, -713.4682617, 699.0614014
9: -333.1286316, 327.7955017, -298.1673584, 293.6198120, -626.7484131, 625.9627686

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7452521, upper bound: 587.7468046
time: 19.10 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7448166, upper bound: 587.7467077
time: 17.39 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -312.4793396, 249.1727448, -348.6863098, 277.8634338, -590.3427734, 597.8590088
1: -263.6935120, 220.6921082, -294.1145630, 246.0266724, -509.7201538, 514.8066406
2: -345.3385315, 224.5350800, -385.1567078, 250.4280548, -595.7665405, 609.6915894
3: -365.3653564, 193.0323639, -407.1756897, 215.0398865, -580.4052734, 600.2080688
4: -336.7408752, 256.6753235, -375.4347229, 286.3746338, -623.1154785, 632.1100464
5: -300.8449097, 233.8077240, -335.5892944, 260.8315735, -561.6764526, 569.3970337
6: -287.8827515, 277.1607666, -320.9473267, 309.0445251, -596.9272461, 598.1080933
7: -313.9259033, 263.1456604, -350.3210449, 293.5184326, -607.4443359, 613.4666748
8: -379.1596985, 260.1344910, -422.7423706, 290.1181335, -669.2778320, 682.8768311
9: -285.5847168, 281.2675171, -318.5613708, 313.4904785, -599.0751343, 599.8288574

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7097801, upper bound: 587.7119912
time: 14.27 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7092671, upper bound: 587.7106089
time: 12.18 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -306.1221619, 244.1371155, -347.5926819, 276.9804688, -583.1026001, 591.7297974
1: -258.3437195, 216.2197571, -293.1323242, 245.2357635, -503.5794678, 509.3520508
2: -338.3315430, 220.0066071, -383.9054871, 249.6025238, -587.9340820, 603.9121094
3: -357.8959961, 189.1206055, -405.8835449, 214.3443909, -572.2402954, 595.0041504
4: -329.8908997, 251.4324951, -374.2347412, 285.4048767, -615.2957764, 625.6672363
5: -294.7322998, 229.0382996, -334.5148926, 259.9424438, -554.6746826, 563.5531616
6: -282.0531311, 271.5390015, -319.9625549, 308.0632629, -590.1163940, 591.5015259
7: -307.5113831, 257.7834167, -349.1604309, 292.5524292, -600.0637207, 606.9437866
8: -371.5134277, 254.9008026, -421.3930664, 289.1515198, -660.6649170, 676.2938232
9: -279.7433472, 275.5548706, -317.5015564, 312.4538269, -592.1971436, 593.0563965

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7096884, upper bound: 587.7116692
time: 9.49 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7088445, upper bound: 587.7088445
time: 10.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.85 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 6, lower bound: -587.7479441, upper bound: 587.7499823
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 6, lower bound: -587.7570502, upper bound: 587.7590029
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 6, lower bound: -587.7452521, upper bound: 587.7468046
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 6, lower bound: -587.7448166, upper bound: 587.7467077
IS_B2_B1_B1, status: Status.VERIFIED, split count: 3, time: 22.85
Output dim: 6, lower bound: -587.7097801, upper bound: 587.7119912
IS_B2_B1_B2, status: Status.VERIFIED, split count: 3, time: 22.85
Output dim: 6, lower bound: -587.7092671, upper bound: 587.7106089
IS_B2_B2_B1, status: Status.VERIFIED, split count: 3, time: 22.85
Output dim: 6, lower bound: -587.7096884, upper bound: 587.7116692
IS_B2_B2_B2, status: Status.VERIFIED, split count: 3, time: 22.85
Output dim: 6, lower bound: -587.7088445, upper bound: 587.7088445

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -327.6079102, 261.1775208, -321.9942017, 256.7422180, -584.3499756, 583.1716309
1: -276.3737793, 231.3455963, -271.7032471, 227.4135742, -503.7873230, 503.0488281
2: -361.9759521, 235.3447418, -355.8101196, 231.3603668, -593.3363037, 591.1547852
3: -383.0221558, 202.3367310, -376.4951172, 198.8957062, -581.9178467, 578.8317871
4: -352.9167480, 269.0776367, -346.9073792, 264.4832458, -617.3999023, 615.9849854
5: -315.3769836, 245.1299591, -310.0080872, 240.9506226, -556.3275757, 555.1380615
6: -301.7168579, 290.4729614, -296.5834351, 285.5453796, -587.2622070, 587.0563965
7: -329.1233826, 275.8897400, -323.5009460, 271.1766968, -600.3000488, 599.3906250
8: -397.3283081, 272.5983887, -390.5945740, 268.0199890, -665.3482666, 663.1929932
9: -299.3767395, 294.7984314, -294.2730408, 289.7997742, -589.1765137, 589.0714722

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7891554, upper bound: 587.7891554
time: 16.98 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7891554, upper bound: 587.7893414
time: 18.53 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -324.9497070, 259.0822144, -325.4350586, 259.4661865, -584.4158936, 584.5172729
1: -274.1781921, 229.4822083, -274.5860901, 229.8243561, -504.0025330, 504.0682983
2: -359.0476074, 233.4457855, -359.5805359, 233.7890778, -592.8366089, 593.0263062
3: -379.9520874, 200.6967468, -380.5217896, 200.9940033, -580.9459839, 581.2183838
4: -350.0805664, 266.8938293, -350.6016541, 267.2912292, -617.3718262, 617.4954834
5: -312.8537598, 243.1492157, -313.3212585, 243.5098572, -556.3635864, 556.4704590
6: -299.2959290, 288.1547241, -299.7429810, 288.5841064, -587.8800049, 587.8977051
7: -326.4486389, 273.6470337, -326.9329529, 274.0535889, -600.5021973, 600.5799561
8: -394.1493530, 270.4316711, -394.7339478, 270.8295593, -664.9788818, 665.1656494
9: -296.9585571, 292.4334717, -297.4002991, 292.8670654, -589.8256226, 589.8337402

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7893414, upper bound: 587.7891952
time: 18.44 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7893414, upper bound: 587.7904224
time: 19.16 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -348.6863098, 277.8634338, -314.5117493, 250.7899170, -599.4761963, 592.3751831
1: -294.1145630, 246.0266724, -265.4018250, 222.1248627, -516.2394409, 511.4284668
2: -385.1567078, 250.4280548, -347.5782166, 225.9791870, -611.1357422, 598.0062256
3: -407.1756897, 215.0398865, -367.7579346, 194.2843628, -601.4600220, 582.7978516
4: -375.4347229, 286.3746338, -338.9311218, 258.3374329, -633.7721558, 625.3057251
5: -335.5892944, 260.8315735, -302.8070374, 235.3281555, -570.9174805, 563.6386108
6: -320.9473267, 309.0445251, -289.7617493, 278.9595642, -599.9068604, 598.8061523
7: -350.3210449, 293.5184326, -315.9624023, 264.8544617, -615.1755371, 609.4808350
8: -422.7423706, 290.1181335, -381.6082764, 261.8005066, -684.5428467, 671.7264404
9: -318.5613708, 313.4904785, -287.4356384, 283.0851135, -601.6464844, 600.9261475

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7185392, upper bound: 587.7195274
time: 19.65 seconds

## Relational analysis of IS_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7148852, upper bound: 587.7167844
time: 15.76 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -347.5926819, 276.9804688, -308.1466675, 245.7482300, -593.3409424, 585.1271362
1: -293.1323242, 245.2357635, -260.0454407, 217.6469116, -510.7792358, 505.2811890
2: -383.9054871, 249.6025238, -340.5622559, 221.4450378, -605.3505249, 590.1647949
3: -405.8835449, 214.3443909, -360.2792053, 190.3678894, -596.2514648, 574.6235352
4: -374.2347412, 285.4048767, -332.0725403, 253.0881653, -627.3228760, 617.4774170
5: -334.5148926, 259.9424438, -296.6869507, 230.5530243, -565.0678101, 556.6292725
6: -319.9625549, 308.0632629, -283.9249268, 273.3306885, -593.2932129, 591.9881592
7: -349.1604309, 292.5524292, -309.5397644, 259.4855957, -608.6459961, 602.0921631
8: -421.3930664, 289.1515198, -373.9524841, 256.5602112, -677.9532471, 663.1040039
9: -317.5015564, 312.4538269, -281.5870972, 277.3652954, -594.8668213, 594.0408936

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7180103, upper bound: 587.7192490
time: 18.60 seconds

## Relational analysis of IS_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7113571, upper bound: 587.7149784
time: 16.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 37.91 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.91
Output dim: 6, lower bound: -587.7891554, upper bound: 587.7891554
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.91
Output dim: 6, lower bound: -587.7891554, upper bound: 587.7893414
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.91
Output dim: 6, lower bound: -587.7893414, upper bound: 587.7891952
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.91
Output dim: 6, lower bound: -587.7893414, upper bound: 587.7904224
IS_B1_A2_A1_A1, status: Status.VERIFIED, split count: 4, time: 37.91
Output dim: 6, lower bound: -587.7185392, upper bound: 587.7195274
IS_B1_A2_A1_A2, status: Status.VERIFIED, split count: 4, time: 37.91
Output dim: 6, lower bound: -587.7148852, upper bound: 587.7167844
IS_B1_A2_A2_A1, status: Status.VERIFIED, split count: 4, time: 37.91
Output dim: 6, lower bound: -587.7180103, upper bound: 587.7192490
IS_B1_A2_A2_A2, status: Status.VERIFIED, split count: 4, time: 37.91
Output dim: 6, lower bound: -587.7113571, upper bound: 587.7149784

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -327.6079102, 261.1775208, -327.6079102, 261.1775208, -588.7852783, 588.7852783
1: -276.3737793, 231.3455963, -276.3737793, 231.3455963, -507.7193604, 507.7193604
2: -361.9759521, 235.3447418, -361.9759521, 235.3447418, -597.3206787, 597.3206787
3: -383.0221558, 202.3367310, -383.0221558, 202.3367310, -585.3588867, 585.3588867
4: -352.9167480, 269.0776367, -352.9167480, 269.0776367, -621.9943848, 621.9943848
5: -315.3769836, 245.1299591, -315.3769836, 245.1299591, -560.5068970, 560.5068970
6: -301.7168579, 290.4729614, -301.7168579, 290.4729614, -592.1898193, 592.1898193
7: -329.1233826, 275.8897400, -329.1233826, 275.8897400, -605.0131226, 605.0131226
8: -397.3283081, 272.5983887, -397.3283081, 272.5983887, -669.9266968, 669.9266968
9: -299.3767395, 294.7984314, -299.3767395, 294.7984314, -594.1751709, 594.1751709

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7121058, upper bound: 587.7161717
time: 15.69 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -587.7082440, upper bound: 587.7082440
time: 11.62 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -327.6079102, 261.1775208, -324.9196472, 259.0589905, -586.6666870, 586.0970459
1: -276.3737793, 231.3455963, -274.1521912, 229.4597321, -505.8334656, 505.4978027
2: -361.9759521, 235.3447418, -359.0156860, 233.4249420, -595.4007568, 594.3604126
3: -383.0221558, 202.3367310, -379.9166870, 200.6783600, -583.7005005, 582.2533569
4: -352.9167480, 269.0776367, -350.0486145, 266.8699951, -619.7867432, 619.1262207
5: -315.3769836, 245.1299591, -312.8253479, 243.1271515, -558.5040283, 557.9553223
6: -301.7168579, 290.4729614, -299.2692871, 288.1284180, -589.8451538, 589.7422485
7: -329.1233826, 275.8897400, -326.4192200, 273.6226196, -602.7459717, 602.3088989
8: -397.3283081, 272.5983887, -394.1150208, 270.4076538, -667.7359619, 666.7133789
9: -299.3767395, 294.7984314, -296.9314575, 292.4072876, -591.7840576, 591.7298584

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7607273, upper bound: 587.7615676
time: 18.45 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7575839, upper bound: 587.7587312
time: 18.74 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -324.9497070, 259.0822144, -327.6079102, 261.1775208, -586.1271973, 586.6898804
1: -274.1781921, 229.4822083, -276.3737793, 231.3455963, -505.5238037, 505.8559875
2: -359.0476074, 233.4457855, -361.9759521, 235.3447418, -594.3923340, 595.4216919
3: -379.9520874, 200.6967468, -383.0221558, 202.3367310, -582.2888184, 583.7188110
4: -350.0805664, 266.8938293, -352.9167480, 269.0776367, -619.1582031, 619.8104858
5: -312.8537598, 243.1492157, -315.3769836, 245.1299591, -557.9837036, 558.5261230
6: -299.2959290, 288.1547241, -301.7168579, 290.4729614, -589.7689209, 589.8715820
7: -326.4486389, 273.6470337, -329.1233826, 275.8897400, -602.3383789, 602.7703857
8: -394.1493530, 270.4316711, -397.3283081, 272.5983887, -666.7477417, 667.7600098
9: -296.9585571, 292.4334717, -299.3767395, 294.7984314, -591.7569580, 591.8101807

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7615676, upper bound: 587.7629089
time: 17.69 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7587312, upper bound: 587.7580540
time: 16.24 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -324.9497070, 259.0822144, -324.9497070, 259.0822144, -584.0317993, 584.0317993
1: -274.1781921, 229.4822083, -274.1781921, 229.4822083, -503.6604004, 503.6604004
2: -359.0476074, 233.4457855, -359.0476074, 233.4457855, -592.4933472, 592.4933472
3: -379.9520874, 200.6967468, -379.9520874, 200.6967468, -580.6486816, 580.6486816
4: -350.0805664, 266.8938293, -350.0805664, 266.8938293, -616.9743652, 616.9743652
5: -312.8537598, 243.1492157, -312.8537598, 243.1492157, -556.0029297, 556.0029297
6: -299.2959290, 288.1547241, -299.2959290, 288.1547241, -587.4506836, 587.4506836
7: -326.4486389, 273.6470337, -326.4486389, 273.6470337, -600.0957031, 600.0957031
8: -394.1493530, 270.4316711, -394.1493530, 270.4316711, -664.5810547, 664.5810547
9: -296.9585571, 292.4334717, -296.9585571, 292.4334717, -589.3920288, 589.3920288

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 136

## Relational analysis of IS_B1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7253741, upper bound: 587.7416625
time: 19.39 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7103275, upper bound: 587.7370358
time: 15.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 35.88 seconds
IS_B1_A1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 35.88
Output dim: 6, lower bound: -587.7121058, upper bound: 587.7161717
IS_B1_A1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 35.88
Output dim: 6, lower bound: -587.7082440, upper bound: 587.7082440
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 35.88
Output dim: 6, lower bound: -587.7607273, upper bound: 587.7615676
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 35.88
Output dim: 6, lower bound: -587.7575839, upper bound: 587.7587312
IS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 35.88
Output dim: 6, lower bound: -587.7615676, upper bound: 587.7629089
IS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 35.88
Output dim: 6, lower bound: -587.7587312, upper bound: 587.7580540
IS_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 35.88
Output dim: 6, lower bound: -587.7253741, upper bound: 587.7416625
IS_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 35.88
Output dim: 6, lower bound: -587.7103275, upper bound: 587.7370358

## BFS IS instance: IS_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -325.3118896, 259.3447571, -323.8524170, 258.2070923, -583.5189209, 583.1971436
1: -274.4511719, 229.7239227, -273.2583313, 228.7059326, -503.1571045, 502.9822388
2: -359.4304810, 233.6932068, -357.8321533, 232.6574860, -592.0879517, 591.5253906
3: -380.3078308, 200.9264679, -378.6549683, 200.0228119, -580.3305054, 579.5814209
4: -350.4007874, 267.1660156, -348.8796082, 265.9816284, -616.3822021, 616.0456543
5: -313.1690979, 243.4059906, -311.7989807, 242.3260651, -555.4949951, 555.2049561
6: -299.5843201, 288.4290161, -298.2778625, 287.1784668, -586.7626953, 586.7069092
7: -326.8046265, 273.9393616, -325.3415833, 272.7160645, -599.5205688, 599.2809448
8: -394.5352783, 270.7072144, -392.8162231, 269.5287170, -664.0639648, 663.5234375
9: -297.2490234, 292.7151184, -295.9424133, 291.4391785, -588.6882324, 588.6574707

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7580540, upper bound: 587.7587312
time: 15.11 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7580540, upper bound: 587.7587312
time: 15.06 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -326.3493042, 260.0880432, -316.3742981, 252.2325287, -578.5817871, 576.4623413
1: -275.2896729, 230.3892822, -266.9809265, 223.4140320, -498.7036743, 497.3701477
2: -360.4375000, 234.2218018, -349.5210571, 227.2595062, -587.6968994, 583.7427979
3: -381.4634094, 201.6155853, -369.8038330, 195.4326019, -576.8959961, 571.4193726
4: -351.2261963, 267.8157654, -340.6564026, 259.7304077, -610.9565430, 608.4721680
5: -314.2194824, 244.0758362, -304.6102905, 236.6946869, -550.9141846, 548.6860352
6: -300.3816833, 289.2800293, -291.3215027, 280.4969482, -580.8786621, 580.6015625
7: -327.7268066, 274.6626892, -317.7638550, 266.3542480, -594.0810547, 592.4265137
8: -395.5231018, 271.4689026, -383.6860962, 263.3399353, -658.8629761, 655.1550293
9: -297.9620972, 293.3948059, -288.9884949, 284.6213989, -582.5834351, 582.3832397

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7580540, upper bound: 587.7587312
time: 15.97 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7580540, upper bound: 587.7587312
time: 15.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 33.00 seconds
IS_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 33.00
Output dim: 6, lower bound: -587.7580540, upper bound: 587.7587312
IS_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 33.00
Output dim: 6, lower bound: -587.7580540, upper bound: 587.7587312
IS_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 33.00
Output dim: 6, lower bound: -587.7580540, upper bound: 587.7587312
IS_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 33.00
Output dim: 6, lower bound: -587.7580540, upper bound: 587.7587312
IS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 33.00
Output dim: 6, lower bound: -587.7615676, upper bound: 587.7629089
IS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 33.00
Output dim: 6, lower bound: -587.7587312, upper bound: 587.7580540
IS_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 33.00
Output dim: 6, lower bound: -587.7253741, upper bound: 587.7416625
IS_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 33.00
Output dim: 6, lower bound: -587.7103275, upper bound: 587.7370358
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=591.3155517578125
rel_dist={6: [-587.7904223681265, 587.7904223711924]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1804.92 seconds
