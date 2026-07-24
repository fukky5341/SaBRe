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
execution time: IAR + LP analysis = 1.07 + 12.71 = 13.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -587.7908408, upper bound: 587.7908408


# Binary Search by BASE starts (time budget: 2686.22 seconds, max iter: 100)

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
Binary search time: 51.32 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2634.90 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7906183, upper bound: 587.7906356
time: 10.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905600, upper bound: 587.7905600
time: 10.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.48 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.48
Output dim: 6, lower bound: -587.7906183, upper bound: 587.7906356
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.48
Output dim: 6, lower bound: -587.7905600, upper bound: 587.7905600

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -316.2442627, 252.1643524, -326.5493774, 260.3517151, -576.5959473, 578.7137451
1: -266.8026123, 223.3411560, -275.5249939, 230.6120300, -497.4146423, 498.8661194
2: -349.4259644, 227.1557770, -360.8080139, 234.5787048, -584.0046387, 587.9638062
3: -369.7535400, 195.3359680, -381.8336182, 201.6814728, -571.4349365, 577.1695557
4: -340.6900635, 259.7447510, -351.7984924, 268.2037659, -608.8937988, 611.5431519
5: -304.4875183, 236.6209106, -314.3969421, 244.3412170, -548.8286743, 551.0178223
6: -291.2841187, 280.4365845, -300.7708435, 289.5719604, -580.8559570, 581.2073975
7: -317.6923523, 266.3090820, -328.0461426, 274.9884033, -592.6806641, 594.3552246
8: -383.5415039, 263.1564026, -396.0783081, 271.7464294, -655.2877808, 659.2347412
9: -289.0002136, 284.6189575, -298.4139099, 293.8655396, -582.8657227, 583.0328369

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7906000, upper bound: 587.7906163
time: 9.66 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
time: 11.62 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -324.2376099, 258.5200195, -327.0875244, 260.7777405, -585.0152588, 585.6073608
1: -273.5869751, 228.9864655, -275.9752808, 230.9897614, -504.5767212, 504.9617004
2: -358.2707520, 232.9467773, -361.3987427, 234.9588470, -593.2296143, 594.3454590
3: -379.1147461, 200.2671814, -382.4662781, 202.0105438, -581.1253052, 582.7334595
4: -349.2968750, 266.3111877, -352.3798218, 268.6433411, -617.9401855, 618.6910400
5: -312.1709595, 242.6206055, -314.9150391, 244.7410431, -556.9119873, 557.5355835
6: -298.6396790, 287.5253296, -301.2674561, 290.0480957, -588.6877441, 588.7926636
7: -325.7356262, 273.0462036, -328.5839539, 275.4401550, -601.1757202, 601.6301270
8: -393.2973633, 269.8595886, -396.7255249, 272.1857910, -665.4831543, 666.5850830
9: -296.3012085, 291.8064270, -298.9044800, 294.3448181, -590.6459961, 590.7109375

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905600, upper bound: 587.7905600
time: 10.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905600, upper bound: 587.7905600
time: 10.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.99 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.99
Output dim: 6, lower bound: -587.7906000, upper bound: 587.7906163
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.99
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.99
Output dim: 6, lower bound: -587.7905600, upper bound: 587.7905600
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.99
Output dim: 6, lower bound: -587.7905600, upper bound: 587.7905600

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -315.9732361, 251.9498291, -321.3295288, 256.2231445, -572.1962280, 573.2793579
1: -266.5755615, 223.1509552, -271.1558228, 226.9517212, -493.5272522, 494.3067627
2: -349.1275940, 226.9632721, -355.0646057, 230.8730927, -580.0006714, 582.0278931
3: -369.4348145, 195.1703186, -375.7022705, 198.4937439, -567.9285278, 570.8725586
4: -340.3992310, 259.5231018, -346.1989441, 263.9368896, -604.3361206, 605.7219849
5: -304.2264709, 236.4190063, -309.3721313, 240.4544220, -544.6809082, 545.7911377
6: -291.0338135, 280.1966553, -295.9510803, 284.9544678, -575.9882202, 576.1477051
7: -317.4206848, 266.0811462, -322.8155212, 270.6006470, -588.0213623, 588.8965454
8: -383.2158203, 262.9358521, -389.8117981, 267.5029907, -650.7188110, 652.7476196
9: -288.7539062, 284.3763123, -293.6719666, 289.1967773, -577.9506226, 578.0482788

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
time: 13.16 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
time: 12.33 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -316.0437012, 252.0055237, -332.2265320, 264.8667908, -580.9104614, 584.2320557
1: -266.6343384, 223.2003021, -280.3401184, 234.6066742, -501.2410278, 503.5404053
2: -349.2051086, 227.0134277, -367.1090698, 238.7130432, -587.9181519, 594.1224976
3: -369.5174561, 195.2131348, -388.3509827, 205.1157074, -574.6331787, 583.5640869
4: -340.4746704, 259.5806580, -357.9318848, 272.8944092, -613.3690796, 617.5125122
5: -304.2942505, 236.4713745, -319.8331604, 248.6227417, -552.9169312, 556.3045654
6: -291.0989990, 280.2589111, -305.9789124, 294.5411072, -585.6399536, 586.2377930
7: -317.4912109, 266.1404419, -333.8062439, 279.7745667, -597.2657471, 599.9465942
8: -383.3001404, 262.9930115, -402.9830017, 276.4886169, -659.7887573, 665.9760132
9: -288.8179626, 284.4392090, -303.6066895, 298.9780273, -587.7959595, 588.0458984

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
time: 9.54 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
time: 10.19 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -324.2376099, 258.5200195, -316.2442627, 252.1643524, -576.4019165, 574.7642212
1: -273.5869751, 228.9864655, -266.8026123, 223.3411560, -496.9280396, 495.7890320
2: -358.2707520, 232.9467773, -349.4259644, 227.1557770, -585.4265137, 582.3727417
3: -379.1147461, 200.2671814, -369.7535400, 195.3359680, -574.4506836, 570.0207520
4: -349.2968750, 266.3111877, -340.6900635, 259.7447510, -609.0415039, 607.0012207
5: -312.1709595, 242.6206055, -304.4875183, 236.6209106, -548.7918701, 547.1081543
6: -298.6396790, 287.5253296, -291.2841187, 280.4365845, -579.0762939, 578.8093262
7: -325.7356262, 273.0462036, -317.6923523, 266.3090820, -592.0446777, 590.7384644
8: -393.2973633, 269.8595886, -383.5415039, 263.1564026, -656.4537354, 653.4010620
9: -296.3012085, 291.8064270, -289.0002136, 284.6189575, -580.9201660, 580.8066406

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905573, upper bound: 587.7905575
time: 11.41 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
time: 10.83 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -324.2376099, 258.5200195, -324.2376099, 258.5200195, -582.7575073, 582.7575073
1: -273.5869751, 228.9864655, -273.5869751, 228.9864655, -502.5733643, 502.5733643
2: -358.2707520, 232.9467773, -358.2707520, 232.9467773, -591.2175293, 591.2175293
3: -379.1147461, 200.2671814, -379.1147461, 200.2671814, -579.3819580, 579.3819580
4: -349.2968750, 266.3111877, -349.2968750, 266.3111877, -615.6080322, 615.6080322
5: -312.1709595, 242.6206055, -312.1709595, 242.6206055, -554.7915649, 554.7915649
6: -298.6396790, 287.5253296, -298.6396790, 287.5253296, -586.1649780, 586.1649780
7: -325.7356262, 273.0462036, -325.7356262, 273.0462036, -598.7817993, 598.7817993
8: -393.2973633, 269.8595886, -393.2973633, 269.8595886, -663.1569824, 663.1569824
9: -296.3012085, 291.8064270, -296.3012085, 291.8064270, -588.1076050, 588.1076050

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905573, upper bound: 587.7905575
time: 9.52 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
time: 10.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.54 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.54
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.54
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.54
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.54
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.54
Output dim: 6, lower bound: -587.7905573, upper bound: 587.7905575
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.54
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.54
Output dim: 6, lower bound: -587.7905573, upper bound: 587.7905575
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.54
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -311.0162354, 248.0300598, -321.3295288, 256.2231445, -567.2393799, 569.3596191
1: -262.4277039, 219.6758575, -271.1558228, 226.9517212, -489.3793640, 490.8316650
2: -343.6754150, 223.4442291, -355.0646057, 230.8730927, -574.5484619, 578.5088501
3: -363.6126709, 192.1442566, -375.7022705, 198.4937439, -562.1063843, 567.8465576
4: -335.0830383, 255.4726868, -346.1989441, 263.9368896, -599.0198975, 601.6716309
5: -299.4560852, 232.7281647, -309.3721313, 240.4544220, -539.9103394, 542.1002808
6: -286.4585571, 275.8131714, -295.9510803, 284.9544678, -571.4130249, 571.7640991
7: -312.4542542, 261.9156799, -322.8155212, 270.6006470, -583.0549316, 584.7311401
8: -377.2666321, 258.9074707, -389.8117981, 267.5029907, -644.7696533, 648.7191162
9: -284.2522888, 279.9438171, -293.6719666, 289.1967773, -573.4489136, 573.6157837

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7906000, upper bound: 587.7906163
time: 10.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7906000, upper bound: 587.7906163
time: 12.32 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -321.0994873, 256.0238647, -321.3295288, 256.2231445, -577.3226318, 577.3533936
1: -270.9159241, 226.7545166, -271.1558228, 226.9517212, -497.8676147, 497.9103088
2: -354.8039246, 230.6958618, -355.0646057, 230.8730927, -585.6770020, 585.7604980
3: -375.2988586, 198.2595673, -375.7022705, 198.4937439, -573.7924805, 573.9617920
4: -345.9213257, 263.7604980, -346.1989441, 263.9368896, -609.8582153, 609.9594727
5: -309.1308899, 240.2902985, -309.3721313, 240.4544220, -549.5852051, 549.6624146
6: -295.7290039, 284.6652222, -295.9510803, 284.9544678, -580.6834717, 580.6162720
7: -322.6116333, 270.4028015, -322.8155212, 270.6006470, -593.2122803, 593.2182617
8: -389.4352112, 267.2122192, -389.8117981, 267.5029907, -656.9382324, 657.0239258
9: -293.4340820, 288.9800720, -293.6719666, 289.1967773, -582.6306763, 582.6520386

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7906000, upper bound: 587.7906163
time: 10.64 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7906000, upper bound: 587.7906163
time: 11.73 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -311.0162354, 248.0300598, -332.2265320, 264.8667908, -575.8830566, 580.2565918
1: -262.4277039, 219.6758575, -280.3401184, 234.6066742, -497.0343323, 500.0159912
2: -343.6754150, 223.4442291, -367.1090698, 238.7130432, -582.3883667, 590.5532837
3: -363.6126709, 192.1442566, -388.3509827, 205.1157074, -568.7283325, 580.4952393
4: -335.0830383, 255.4726868, -357.9318848, 272.8944092, -607.9774170, 613.4045410
5: -299.4560852, 232.7281647, -319.8331604, 248.6227417, -548.0787964, 552.5612793
6: -286.4585571, 275.8131714, -305.9789124, 294.5411072, -580.9995117, 581.7919312
7: -312.4542542, 261.9156799, -333.8062439, 279.7745667, -592.2288208, 595.7218628
8: -377.2666321, 258.9074707, -402.9830017, 276.4886169, -653.7552490, 661.8904419
9: -284.2522888, 279.9438171, -303.6066895, 298.9780273, -583.2302856, 583.5504150

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
time: 9.97 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
time: 10.63 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -321.0994873, 256.0238647, -332.2265320, 264.8667908, -585.9663086, 588.2503662
1: -270.9159241, 226.7545166, -280.3401184, 234.6066742, -505.5225830, 507.0946350
2: -354.8039246, 230.6958618, -367.1090698, 238.7130432, -593.5169067, 597.8049316
3: -375.2988586, 198.2595673, -388.3509827, 205.1157074, -580.4144287, 586.6104736
4: -345.9213257, 263.7604980, -357.9318848, 272.8944092, -618.8157349, 621.6923828
5: -309.1308899, 240.2902985, -319.8331604, 248.6227417, -557.7536621, 560.1234741
6: -295.7290039, 284.6652222, -305.9789124, 294.5411072, -590.2699585, 590.6441040
7: -322.6116333, 270.4028015, -333.8062439, 279.7745667, -602.3861694, 604.2089844
8: -389.4352112, 267.2122192, -402.9830017, 276.4886169, -665.9238281, 670.1951294
9: -293.4340820, 288.9800720, -303.6066895, 298.9780273, -592.4120483, 592.5867310

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
time: 10.14 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
time: 12.13 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -319.0052795, 254.3818665, -315.9732361, 251.9498291, -570.9550781, 570.3550415
1: -269.2081909, 225.3180695, -266.5755615, 223.1509552, -492.3591309, 491.8935852
2: -352.5144348, 229.2324066, -349.1275940, 226.9632721, -579.4777222, 578.3599243
3: -372.9690552, 197.0719757, -369.4348145, 195.1703186, -568.1392822, 566.5067749
4: -343.6843872, 262.0352783, -340.3992310, 259.5231018, -603.2073364, 602.4345093
5: -307.1348267, 238.7250519, -304.2264709, 236.4190063, -543.5538330, 542.9515381
6: -293.8093567, 282.8973083, -291.0338135, 280.1966553, -574.0059814, 573.9310913
7: -320.4930725, 268.6490784, -317.4206848, 266.0811462, -586.5740967, 586.0697632
8: -387.0165710, 265.6063538, -383.2158203, 262.9358521, -649.9523926, 648.8220825
9: -291.5490112, 287.1274719, -288.7539062, 284.3763123, -575.9252930, 575.8813477

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7906163, upper bound: 587.7905996
time: 9.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7906163, upper bound: 587.7905996
time: 9.86 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -329.8842468, 263.0115967, -316.0437012, 252.0055237, -581.8897705, 579.0552979
1: -278.3775940, 232.9605408, -266.6343384, 223.2003021, -501.5778809, 499.5948792
2: -364.5394592, 237.0601654, -349.2051086, 227.0134277, -591.5528564, 586.2652588
3: -385.5976562, 203.6827393, -369.5174561, 195.2131348, -580.8106689, 573.2001953
4: -355.3976135, 270.9781494, -340.4746704, 259.5806580, -614.9780884, 611.4527588
5: -317.5791321, 246.8801422, -304.2942505, 236.4713745, -554.0505371, 551.1743164
6: -303.8206482, 292.4688110, -291.0989990, 280.2589111, -584.0795288, 583.5678101
7: -331.4658813, 277.8080444, -317.4912109, 266.1404419, -597.6062012, 595.2992554
8: -400.1672058, 274.5773621, -383.3001404, 262.9930115, -663.1602173, 657.8774414
9: -301.4669495, 296.8927917, -288.8179626, 284.4392090, -585.9061279, 585.7107544

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7906163, upper bound: 587.7905996
time: 9.57 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7906163, upper bound: 587.7905996
time: 11.65 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -319.0052795, 254.3818665, -323.9680786, 258.3066101, -577.3118896, 578.3499756
1: -269.2081909, 225.3180695, -273.3611755, 228.7972870, -498.0054626, 498.6791992
2: -352.5144348, 229.2324066, -357.9740601, 232.7553864, -585.2697754, 587.2063599
3: -372.9690552, 197.0719757, -378.7979126, 200.1024323, -573.0714111, 575.8698730
4: -343.6843872, 262.0352783, -349.0075989, 266.0907593, -609.7751465, 611.0428467
5: -307.1348267, 238.7250519, -311.9113464, 242.4198761, -549.5546875, 550.6363525
6: -293.8093567, 282.8973083, -298.3906860, 287.2866211, -581.0959473, 581.2879639
7: -320.4930725, 268.6490784, -325.4655457, 272.8195190, -593.3126221, 594.1146240
8: -387.0165710, 265.6063538, -392.9736023, 269.6401978, -656.6567383, 658.5798950
9: -291.5490112, 287.1274719, -296.0562439, 291.5652161, -583.1142578, 583.1835938

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
time: 9.73 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
time: 11.37 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -329.8842468, 263.0115967, -324.0376892, 258.3615417, -588.2457886, 587.0493164
1: -278.3775940, 232.9605408, -273.4191895, 228.8459778, -507.2235718, 506.3797302
2: -364.5394592, 237.0601654, -358.0505066, 232.8049011, -597.3443604, 595.1106567
3: -385.5976562, 203.6827393, -378.8794861, 200.1446686, -585.7421875, 582.5621948
4: -355.3976135, 270.9781494, -349.0820312, 266.1475220, -621.5451660, 620.0601196
5: -317.5791321, 246.8801422, -311.9782715, 242.4715424, -560.0506592, 558.8583984
6: -303.8206482, 292.4688110, -298.4551086, 287.3480835, -591.1687012, 590.9239502
7: -331.4658813, 277.8080444, -325.5351562, 272.8779602, -604.3437500, 603.3432007
8: -400.1672058, 274.5773621, -393.0567627, 269.6966248, -669.8638306, 667.6340942
9: -301.4669495, 296.8927917, -296.1194763, 291.6271973, -593.0941162, 593.0121460

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
time: 9.82 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
time: 10.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.58 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7906000, upper bound: 587.7906163
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7906000, upper bound: 587.7906163
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7906000, upper bound: 587.7906163
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7906000, upper bound: 587.7906163
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7905996, upper bound: 587.7906163
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7906163, upper bound: 587.7905996
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7906163, upper bound: 587.7905996
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7906163, upper bound: 587.7905996
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7906163, upper bound: 587.7905996
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -311.0162354, 248.0300598, -311.0162354, 248.0300598, -559.0462646, 559.0462646
1: -262.4277039, 219.6758575, -262.4277039, 219.6758575, -482.1035156, 482.1035156
2: -343.6754150, 223.4442291, -343.6754150, 223.4442291, -567.1196289, 567.1196289
3: -363.6126709, 192.1442566, -363.6126709, 192.1442566, -555.7569580, 555.7569580
4: -335.0830383, 255.4726868, -335.0830383, 255.4726868, -590.5556641, 590.5556641
5: -299.4560852, 232.7281647, -299.4560852, 232.7281647, -532.1841431, 532.1841431
6: -286.4585571, 275.8131714, -286.4585571, 275.8131714, -562.2717285, 562.2717285
7: -312.4542542, 261.9156799, -312.4542542, 261.9156799, -574.3698730, 574.3698730
8: -377.2666321, 258.9074707, -377.2666321, 258.9074707, -636.1740723, 636.1740723
9: -284.2522888, 279.9438171, -284.2522888, 279.9438171, -564.1960449, 564.1960449

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7882758, upper bound: 587.7861574
time: 11.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854338, upper bound: 587.7853547
time: 9.13 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -311.0162354, 248.0300598, -319.0052795, 254.3818665, -565.3980713, 567.0353394
1: -262.4277039, 219.6758575, -269.2081909, 225.3180695, -487.7456970, 488.8840332
2: -343.6754150, 223.4442291, -352.5144348, 229.2324066, -572.9077148, 575.9586182
3: -363.6126709, 192.1442566, -372.9690552, 197.0719757, -560.6846313, 565.1132812
4: -335.0830383, 255.4726868, -343.6843872, 262.0352783, -597.1182861, 599.1570435
5: -299.4560852, 232.7281647, -307.1348267, 238.7250519, -538.1810303, 539.8629761
6: -286.4585571, 275.8131714, -293.8093567, 282.8973083, -569.3558350, 569.6224365
7: -312.4542542, 261.9156799, -320.4930725, 268.6490784, -581.1033325, 582.4086914
8: -377.2666321, 258.9074707, -387.0165710, 265.6063538, -642.8729858, 645.9239502
9: -284.2522888, 279.9438171, -291.5490112, 287.1274719, -571.3796997, 571.4927979

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7882758, upper bound: 587.7861574
time: 11.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854338, upper bound: 587.7853547
time: 9.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -321.0994873, 256.0238647, -311.0162354, 248.0300598, -569.1295166, 567.0401001
1: -270.9159241, 226.7545166, -262.4277039, 219.6758575, -490.5917664, 489.1821594
2: -354.8039246, 230.6958618, -343.6754150, 223.4442291, -578.2481689, 574.3712769
3: -375.2988586, 198.2595673, -363.6126709, 192.1442566, -567.4431152, 561.8721924
4: -345.9213257, 263.7604980, -335.0830383, 255.4726868, -601.3939209, 598.8435059
5: -309.1308899, 240.2902985, -299.4560852, 232.7281647, -541.8589478, 539.7463379
6: -295.7290039, 284.6652222, -286.4585571, 275.8131714, -571.5421143, 571.1237793
7: -322.6116333, 270.4028015, -312.4542542, 261.9156799, -584.5272217, 582.8569946
8: -389.4352112, 267.2122192, -377.2666321, 258.9074707, -648.3426514, 644.4788818
9: -293.4340820, 288.9800720, -284.2522888, 279.9438171, -573.3778076, 573.2323608

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7882584, upper bound: 587.7861322
time: 11.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7853972, upper bound: 587.7853111
time: 10.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -321.0994873, 256.0238647, -319.0052795, 254.3818665, -575.4813232, 575.0291748
1: -270.9159241, 226.7545166, -269.2081909, 225.3180695, -496.2339783, 495.9626770
2: -354.8039246, 230.6958618, -352.5144348, 229.2324066, -584.0362549, 583.2103271
3: -375.2988586, 198.2595673, -372.9690552, 197.0719757, -572.3707886, 571.2283936
4: -345.9213257, 263.7604980, -343.6843872, 262.0352783, -607.9565430, 607.4448853
5: -309.1308899, 240.2902985, -307.1348267, 238.7250519, -547.8558350, 547.4251099
6: -295.7290039, 284.6652222, -293.8093567, 282.8973083, -578.6263428, 578.4746094
7: -322.6116333, 270.4028015, -320.4930725, 268.6490784, -591.2607422, 590.8958130
8: -389.4352112, 267.2122192, -387.0165710, 265.6063538, -655.0415649, 654.2286987
9: -293.4340820, 288.9800720, -291.5490112, 287.1274719, -580.5614624, 580.5290527

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7882584, upper bound: 587.7861322
time: 11.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7853972, upper bound: 587.7853111
time: 9.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -311.0162354, 248.0300598, -321.0994873, 256.0238647, -567.0401001, 569.1295166
1: -262.4277039, 219.6758575, -270.9159241, 226.7545166, -489.1821594, 490.5917664
2: -343.6754150, 223.4442291, -354.8039246, 230.6958618, -574.3712769, 578.2481689
3: -363.6126709, 192.1442566, -375.2988586, 198.2595673, -561.8721924, 567.4431152
4: -335.0830383, 255.4726868, -345.9213257, 263.7604980, -598.8435059, 601.3939209
5: -299.4560852, 232.7281647, -309.1308899, 240.2902985, -539.7463379, 541.8590088
6: -286.4585571, 275.8131714, -295.7290039, 284.6652222, -571.1237793, 571.5421143
7: -312.4542542, 261.9156799, -322.6116333, 270.4028015, -582.8569946, 584.5272217
8: -377.2666321, 258.9074707, -389.4352112, 267.2122192, -644.4788818, 648.3426514
9: -284.2522888, 279.9438171, -293.4340820, 288.9800720, -573.2323608, 573.3778076

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7882661, upper bound: 587.7861418
time: 12.24 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854000, upper bound: 587.7853312
time: 11.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -311.0162354, 248.0300598, -329.8842468, 263.0115967, -574.0278320, 577.9143066
1: -262.4277039, 219.6758575, -278.3775940, 232.9605408, -495.3882141, 498.0534363
2: -343.6754150, 223.4442291, -364.5394592, 237.0601654, -580.7355957, 587.9837036
3: -363.6126709, 192.1442566, -385.5976562, 203.6827393, -567.2954102, 577.7418823
4: -335.0830383, 255.4726868, -355.3976135, 270.9781494, -606.0610962, 610.8701782
5: -299.4560852, 232.7281647, -317.5791321, 246.8801422, -546.3361206, 550.3073120
6: -286.4585571, 275.8131714, -303.8206482, 292.4688110, -578.9273682, 579.6336670
7: -312.4542542, 261.9156799, -331.4658813, 277.8080444, -590.2623291, 593.3814697
8: -377.2666321, 258.9074707, -400.1672058, 274.5773621, -651.8439941, 659.0746460
9: -284.2522888, 279.9438171, -301.4669495, 296.8927917, -581.1450806, 581.4107666

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7882661, upper bound: 587.7861419
time: 12.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854000, upper bound: 587.7853312
time: 10.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -321.0994873, 256.0238647, -321.0994873, 256.0238647, -577.1233521, 577.1233521
1: -270.9159241, 226.7545166, -270.9159241, 226.7545166, -497.6704102, 497.6704102
2: -354.8039246, 230.6958618, -354.8039246, 230.6958618, -585.4997559, 585.4997559
3: -375.2988586, 198.2595673, -375.2988586, 198.2595673, -573.5582275, 573.5582275
4: -345.9213257, 263.7604980, -345.9213257, 263.7604980, -609.6817627, 609.6817627
5: -309.1308899, 240.2902985, -309.1308899, 240.2902985, -549.4211426, 549.4212036
6: -295.7290039, 284.6652222, -295.7290039, 284.6652222, -580.3942261, 580.3942261
7: -322.6116333, 270.4028015, -322.6116333, 270.4028015, -593.0143433, 593.0143433
8: -389.4352112, 267.2122192, -389.4352112, 267.2122192, -656.6474609, 656.6474609
9: -293.4340820, 288.9800720, -293.4340820, 288.9800720, -582.4141846, 582.4141846

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7882554, upper bound: 587.7861231
time: 10.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7853822, upper bound: 587.7853073
time: 9.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -321.0994873, 256.0238647, -329.8842468, 263.0115967, -584.1110840, 585.9080811
1: -270.9159241, 226.7545166, -278.3775940, 232.9605408, -503.8764648, 505.1320801
2: -354.8039246, 230.6958618, -364.5394592, 237.0601654, -591.8640747, 595.2353516
3: -375.2988586, 198.2595673, -385.5976562, 203.6827393, -578.9815063, 583.8569946
4: -345.9213257, 263.7604980, -355.3976135, 270.9781494, -616.8993530, 619.1580811
5: -309.1308899, 240.2902985, -317.5791321, 246.8801422, -556.0109253, 557.8694458
6: -295.7290039, 284.6652222, -303.8206482, 292.4688110, -588.1978149, 588.4858398
7: -322.6116333, 270.4028015, -331.4658813, 277.8080444, -600.4196777, 601.8685913
8: -389.4352112, 267.2122192, -400.1672058, 274.5773621, -664.0125732, 667.3793335
9: -293.4340820, 288.9800720, -301.4669495, 296.8927917, -590.3269043, 590.4470215

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7882554, upper bound: 587.7861231
time: 11.18 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7853822, upper bound: 587.7853073
time: 10.99 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -319.0052795, 254.3818665, -311.0162354, 248.0300598, -567.0353394, 565.3980713
1: -269.2081909, 225.3180695, -262.4277039, 219.6758575, -488.8840332, 487.7456970
2: -352.5144348, 229.2324066, -343.6754150, 223.4442291, -575.9586182, 572.9077148
3: -372.9690552, 197.0719757, -363.6126709, 192.1442566, -565.1132812, 560.6846313
4: -343.6843872, 262.0352783, -335.0830383, 255.4726868, -599.1570435, 597.1182861
5: -307.1348267, 238.7250519, -299.4560852, 232.7281647, -539.8629761, 538.1810303
6: -293.8093567, 282.8973083, -286.4585571, 275.8131714, -569.6224365, 569.3558350
7: -320.4930725, 268.6490784, -312.4542542, 261.9156799, -582.4086914, 581.1033325
8: -387.0165710, 265.6063538, -377.2666321, 258.9074707, -645.9239502, 642.8729858
9: -291.5490112, 287.1274719, -284.2522888, 279.9438171, -571.4927979, 571.3796997

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7881051, upper bound: 587.7861252
time: 11.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7853111, upper bound: 587.7853972
time: 10.20 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -319.0052795, 254.3818665, -321.0994873, 256.0238647, -575.0291748, 575.4813232
1: -269.2081909, 225.3180695, -270.9159241, 226.7545166, -495.9626770, 496.2339783
2: -352.5144348, 229.2324066, -354.8039246, 230.6958618, -583.2103271, 584.0362549
3: -372.9690552, 197.0719757, -375.2988586, 198.2595673, -571.2283936, 572.3707886
4: -343.6843872, 262.0352783, -345.9213257, 263.7604980, -607.4448853, 607.9565430
5: -307.1348267, 238.7250519, -309.1308899, 240.2902985, -547.4251099, 547.8558960
6: -293.8093567, 282.8973083, -295.7290039, 284.6652222, -578.4746094, 578.6263428
7: -320.4930725, 268.6490784, -322.6116333, 270.4028015, -590.8958130, 591.2607422
8: -387.0165710, 265.6063538, -389.4352112, 267.2122192, -654.2286987, 655.0415649
9: -291.5490112, 287.1274719, -293.4340820, 288.9800720, -580.5290527, 580.5614624

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7881051, upper bound: 587.7861252
time: 12.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7853111, upper bound: 587.7853972
time: 9.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -329.8842468, 263.0115967, -311.0162354, 248.0300598, -577.9143066, 574.0278320
1: -278.3775940, 232.9605408, -262.4277039, 219.6758575, -498.0534363, 495.3882141
2: -364.5394592, 237.0601654, -343.6754150, 223.4442291, -587.9837036, 580.7355957
3: -385.5976562, 203.6827393, -363.6126709, 192.1442566, -577.7418823, 567.2954102
4: -355.3976135, 270.9781494, -335.0830383, 255.4726868, -610.8701782, 606.0610962
5: -317.5791321, 246.8801422, -299.4560852, 232.7281647, -550.3073120, 546.3361206
6: -303.8206482, 292.4688110, -286.4585571, 275.8131714, -579.6336670, 578.9273682
7: -331.4658813, 277.8080444, -312.4542542, 261.9156799, -593.3814697, 590.2623291
8: -400.1672058, 274.5773621, -377.2666321, 258.9074707, -659.0746460, 651.8439941
9: -301.4669495, 296.8927917, -284.2522888, 279.9438171, -581.4107666, 581.1450806

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7881051, upper bound: 587.7861242
time: 10.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7853073, upper bound: 587.7853822
time: 10.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.41 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7882758, upper bound: 587.7861574
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7854338, upper bound: 587.7853547
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7882758, upper bound: 587.7861574
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7854338, upper bound: 587.7853547
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7882584, upper bound: 587.7861322
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7853972, upper bound: 587.7853111
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7882584, upper bound: 587.7861322
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7853972, upper bound: 587.7853111
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7882661, upper bound: 587.7861418
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7854000, upper bound: 587.7853312
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7882661, upper bound: 587.7861419
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7854000, upper bound: 587.7853312
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7882554, upper bound: 587.7861231
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7853822, upper bound: 587.7853073
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7882554, upper bound: 587.7861231
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7853822, upper bound: 587.7853073
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7881051, upper bound: 587.7861252
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7853111, upper bound: 587.7853972
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7881051, upper bound: 587.7861252
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7853111, upper bound: 587.7853972
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7881051, upper bound: 587.7861242
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 6, lower bound: -587.7853073, upper bound: 587.7853822
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 6, lower bound: -587.7906163, upper bound: 587.7905996
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 6, lower bound: -587.7905563, upper bound: 587.7905563
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=591.3155517578125
rel_dist={6: [-587.7907620297522, 587.7907620249766]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7904614, upper bound: 587.7904659
time: 11.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7904191, upper bound: 587.7904191
time: 11.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 23.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 23.26
Output dim: 6, lower bound: -587.7904614, upper bound: 587.7904659
IS_A2, status: Status.UNKNOWN, split count: 1, time: 23.26
Output dim: 6, lower bound: -587.7904191, upper bound: 587.7904191

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -316.2442627, 252.1643524, -324.9513855, 259.0860901, -575.3302612, 577.1157227
1: -266.8026123, 223.3411560, -274.1873779, 229.4898682, -496.2924805, 497.5284729
2: -349.4259644, 227.1557770, -359.0537720, 233.4488373, -582.8747559, 586.2095337
3: -369.7535400, 195.3359680, -379.9538269, 200.7043152, -570.4578247, 575.2897949
4: -340.6900635, 259.7447510, -350.0717773, 266.8978271, -607.5878906, 609.8164062
5: -304.4875183, 236.6209106, -312.8588562, 243.1543579, -547.6418457, 549.4797363
6: -291.2841187, 280.4365845, -299.2957153, 288.1577759, -579.4417114, 579.7322998
7: -317.6923523, 266.3090820, -326.4495544, 273.6469727, -591.3391113, 592.7586060
8: -383.5415039, 263.1564026, -394.1549072, 270.4412842, -653.9827881, 657.3112793
9: -289.0002136, 284.6189575, -296.9569702, 292.4416199, -581.4418335, 581.5759277

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7904490, upper bound: 587.7904538
time: 11.13 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7904488, upper bound: 587.7904537
time: 11.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -324.2376099, 258.5200195, -326.0832825, 259.9822693, -584.2197876, 584.6032104
1: -273.5869751, 228.9864655, -275.1338196, 230.2839355, -503.8709106, 504.1202393
2: -358.2707520, 232.9467773, -360.2964783, 234.2499390, -592.5206909, 593.2432861
3: -379.1147461, 200.2671814, -381.2854614, 201.3961182, -580.5108643, 581.5526123
4: -349.2968750, 266.3111877, -351.2934265, 267.8216553, -617.1184692, 617.6046143
5: -312.1709595, 242.6206055, -313.9479675, 243.9937897, -556.1647339, 556.5686035
6: -298.6396790, 287.5253296, -300.3414612, 289.1591187, -587.7988281, 587.8668213
7: -325.7356262, 273.0462036, -327.5801392, 274.5965881, -600.3321533, 600.6262817
8: -393.2973633, 269.8595886, -395.5176697, 271.3659973, -664.6633301, 665.3772583
9: -296.3012085, 291.8064270, -297.9872131, 293.4503784, -589.7515869, 589.7936401

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7904183, upper bound: 587.7904183
time: 12.43 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7904180, upper bound: 587.7904180
time: 12.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.92 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.92
Output dim: 6, lower bound: -587.7904490, upper bound: 587.7904538
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.92
Output dim: 6, lower bound: -587.7904488, upper bound: 587.7904537
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.92
Output dim: 6, lower bound: -587.7904183, upper bound: 587.7904183
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.92
Output dim: 6, lower bound: -587.7904180, upper bound: 587.7904180

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -314.1911926, 250.5396729, -319.7440796, 254.9675446, -569.1586304, 570.2837524
1: -265.0828857, 221.9005890, -269.8290100, 225.8386078, -490.9214478, 491.7295837
2: -347.1661987, 225.6976471, -353.3245239, 229.7522278, -576.9184570, 579.0221558
3: -367.3399963, 194.0812836, -373.8373718, 197.5243683, -564.8643799, 567.9186401
4: -338.4870911, 258.0657043, -344.4859314, 262.6417236, -601.1287842, 602.5516357
5: -302.5102539, 235.0914917, -307.8463135, 239.2769623, -541.7872314, 542.9378052
6: -289.3880310, 278.6190796, -294.4880371, 283.5515137, -572.9395752, 573.1071167
7: -315.6345825, 264.5823364, -321.2317505, 269.2701416, -584.9047241, 585.8140259
8: -381.0749207, 261.4859314, -387.9038086, 266.2083130, -647.2832031, 649.3897705
9: -287.1345825, 282.7812500, -292.2268677, 287.7843323, -574.9188843, 575.0081177

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7870275, upper bound: 587.7856215
time: 12.47 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7852296, upper bound: 587.7851791
time: 11.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -314.7734375, 250.9994659, -330.6510925, 263.6190491, -578.3924561, 581.6505127
1: -265.5685730, 222.3080902, -279.0215454, 233.5005951, -499.0691528, 501.3295898
2: -347.8062744, 226.1117096, -365.3798523, 237.5988007, -585.4050903, 591.4915771
3: -368.0226440, 194.4348755, -386.4981079, 204.1524353, -572.1750488, 580.9329224
4: -339.1100769, 258.5408325, -356.2297974, 271.6073914, -610.7173462, 614.7706299
5: -303.0699158, 235.5240936, -318.3169861, 247.4525757, -550.5224609, 553.8409424
6: -289.9265747, 279.1335754, -304.5247192, 293.1470947, -583.0736084, 583.6582031
7: -316.2175598, 265.0715942, -332.2324524, 278.4522705, -594.6697998, 597.3040771
8: -381.7714539, 261.9579468, -401.0868835, 275.2019348, -656.9733887, 663.0447388
9: -287.6632690, 283.3010254, -302.1708069, 297.5744324, -585.2376709, 585.4718018

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7870275, upper bound: 587.7856162
time: 12.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7852230, upper bound: 587.7851765
time: 9.75 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -322.1954956, 256.9038391, -320.8558350, 255.8477173, -578.0432129, 577.7596436
1: -271.8764343, 227.5534973, -270.7584229, 226.6184692, -498.4948425, 498.3119202
2: -356.0228882, 231.4965820, -354.5449524, 230.5390015, -586.5617676, 586.0415039
3: -376.7144165, 199.0189819, -375.1452942, 198.2037811, -574.9180908, 574.1643066
4: -347.1052856, 264.6411438, -345.6857910, 263.5488892, -610.6541748, 610.3267822
5: -310.2040710, 241.0994263, -308.9160461, 240.1014557, -550.3054810, 550.0155029
6: -296.7532959, 285.7173157, -295.5149841, 284.5350342, -581.2883301, 581.2322998
7: -323.6889648, 271.3287048, -322.3420410, 270.2026978, -593.8916626, 593.6707764
8: -390.8439331, 268.1978455, -389.2422485, 267.1165466, -657.9604492, 657.4399414
9: -294.4454041, 289.9786987, -293.2385864, 288.7751465, -583.2204590, 583.2172852

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7869106, upper bound: 587.7855801
time: 13.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7851724, upper bound: 587.7851677
time: 11.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -322.7714844, 257.3584900, -331.7453613, 264.4855957, -587.2570190, 589.1036377
1: -272.3566895, 227.9564819, -279.9366150, 234.2682800, -506.6249390, 507.8930664
2: -356.6558228, 231.9060059, -366.5813293, 238.3739166, -595.0296631, 598.4872437
3: -377.3893127, 199.3684998, -387.7855530, 204.8211823, -582.2103882, 587.1539917
4: -347.7215576, 265.1109009, -357.4105835, 272.5003052, -620.2218628, 622.5214844
5: -310.7577820, 241.5271301, -319.3700867, 248.2643127, -559.0220947, 560.8971558
6: -297.2862244, 286.2261658, -305.5359802, 294.1153870, -591.4016113, 591.7620239
7: -324.2651062, 271.8123779, -333.3253479, 279.3704224, -603.6354980, 605.1376953
8: -391.5326233, 268.6644897, -402.4048767, 276.0961609, -667.6287231, 671.0693359
9: -294.9681396, 290.4921570, -303.1664124, 298.5498657, -593.5178833, 593.6585693

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7869039, upper bound: 587.7855643
time: 11.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7851642, upper bound: 587.7851642
time: 10.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.87 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 6, lower bound: -587.7870275, upper bound: 587.7856215
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 6, lower bound: -587.7852296, upper bound: 587.7851791
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 6, lower bound: -587.7870275, upper bound: 587.7856162
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 6, lower bound: -587.7852230, upper bound: 587.7851765
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 6, lower bound: -587.7869106, upper bound: 587.7855801
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 6, lower bound: -587.7851724, upper bound: 587.7851677
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 6, lower bound: -587.7869039, upper bound: 587.7855643
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 6, lower bound: -587.7851642, upper bound: 587.7851642

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -312.0144348, 248.8218842, -319.5351868, 254.8026886, -566.8170776, 568.3569946
1: -263.2631226, 220.3704681, -269.6545105, 225.6917877, -488.9548340, 490.0249023
2: -344.7662354, 224.1560516, -353.0942383, 229.6043243, -574.3705444, 577.2503052
3: -364.7798462, 192.7485046, -373.5918274, 197.3964539, -562.1762695, 566.3403320
4: -336.1286011, 256.2905579, -344.2595520, 262.4715271, -598.5999756, 600.5501099
5: -300.4183044, 233.4811096, -307.6457520, 239.1224365, -539.5407715, 541.1267700
6: -287.3707581, 276.6928711, -294.2945557, 283.3667297, -570.7374878, 570.9874268
7: -313.4563904, 262.7565613, -321.0227051, 269.0950012, -582.5513916, 583.7792969
8: -378.4530640, 259.7096252, -387.6522522, 266.0377808, -644.4908447, 647.3618774
9: -285.1525269, 280.8389893, -292.0367126, 287.5979919, -572.7503052, 572.8757324

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854149, upper bound: 587.7837974
time: 12.13 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7870275, upper bound: 587.7856215
time: 13.86 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -323.5600586, 257.9057922, -318.1656494, 253.7205505, -577.2805786, 576.0714111
1: -272.9087524, 228.4176941, -268.5098572, 224.7269287, -497.6356201, 496.9275513
2: -357.4205627, 232.4410553, -351.5818787, 228.6320801, -586.0525513, 584.0229492
3: -378.2931213, 199.8790741, -371.9839478, 196.5598145, -574.8529053, 571.8630371
4: -348.4481506, 265.6892090, -342.7744141, 261.3553772, -609.8035278, 608.4635010
5: -311.4271851, 242.0747223, -306.3319702, 238.1076660, -549.5347290, 548.4066772
6: -297.9494019, 286.8301697, -293.0308228, 282.1540222, -580.1033936, 579.8609619
7: -325.0664062, 272.4553223, -319.6528320, 267.9468994, -593.0133057, 592.1081543
8: -392.4300537, 269.1462097, -386.0043335, 264.9192505, -657.3492432, 655.1505127
9: -295.6380310, 291.1007385, -290.7901611, 286.3756104, -582.0135498, 581.8908691

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7840322, upper bound: 587.7834056
time: 12.51 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7852296, upper bound: 587.7851791
time: 13.12 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -312.5848999, 249.2723236, -330.4422302, 263.4542847, -576.0391846, 579.7144775
1: -263.7385254, 220.7693787, -278.8470764, 233.3538361, -497.0923462, 499.6164246
2: -345.3930664, 224.5617828, -365.1497192, 237.4509583, -582.8439941, 589.7114868
3: -365.4483643, 193.0951080, -386.2526855, 204.0245514, -569.4729004, 579.3477783
4: -336.7388916, 256.7554626, -356.0035400, 271.4373169, -608.1762085, 612.7590332
5: -300.9663391, 233.9047699, -318.1164246, 247.2980804, -548.2644043, 552.0211792
6: -287.8980713, 277.1966553, -304.3312378, 292.9623413, -580.8604126, 581.5278931
7: -314.0272827, 263.2355957, -332.0235291, 278.2771912, -592.3044434, 595.2590942
8: -379.1351013, 260.1721497, -400.8354492, 275.0315552, -654.1666260, 661.0075684
9: -285.6701660, 281.3478394, -301.9808044, 297.3882446, -583.0582886, 583.3286133

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854149, upper bound: 587.7837974
time: 13.41 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7870275, upper bound: 587.7856162
time: 12.82 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -324.1157532, 258.3441467, -329.0674744, 262.3679810, -586.4837646, 587.4116211
1: -273.3712463, 228.8059540, -277.6980896, 232.3853760, -505.7565918, 506.5040283
2: -358.0305481, 232.8361206, -363.6318970, 236.4753571, -594.5059204, 596.4680176
3: -378.9434814, 200.2160645, -384.6388245, 203.1844635, -582.1279297, 584.8548584
4: -349.0422363, 266.1417236, -354.5125122, 270.3168945, -619.3591309, 620.6542358
5: -311.9607544, 242.4869232, -316.7979126, 246.2793274, -558.2400513, 559.2847900
6: -298.4628296, 287.3201599, -303.0624695, 291.7449951, -590.2078247, 590.3825073
7: -325.6219482, 272.9214172, -330.6482849, 277.1247253, -602.7467041, 603.5697021
8: -393.0936584, 269.5960083, -399.1813354, 273.9085693, -667.0021362, 668.7773438
9: -296.1416321, 291.5957947, -300.7293701, 296.1612854, -592.3029175, 592.3251343

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7840310, upper bound: 587.7834056
time: 13.17 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7840310, upper bound: 587.7851765
time: 12.29 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -320.0429688, 255.2050018, -320.6472473, 255.6831665, -575.7261353, 575.8522339
1: -270.0769348, 226.0400543, -270.5842285, 226.4718628, -496.5487061, 496.6242676
2: -353.6497192, 229.9722137, -354.3150330, 230.3912964, -584.0410156, 584.2872314
3: -374.1829529, 197.7013702, -374.9001160, 198.0761108, -572.2590332, 572.6014404
4: -344.7728577, 262.8860474, -345.4598083, 263.3789978, -608.1518555, 608.3458252
5: -308.1354675, 239.5068665, -308.7157288, 239.9471893, -548.0826416, 548.2225952
6: -294.7587891, 283.8124695, -295.3218079, 284.3505554, -579.1093140, 579.1342773
7: -321.5348511, 269.5232239, -322.1333923, 270.0278931, -591.5627441, 591.6566162
8: -388.2513123, 266.4415588, -388.9911499, 266.9464111, -655.1977539, 655.4325562
9: -292.4851379, 288.0578308, -293.0487671, 288.5891418, -581.0742798, 581.1065674

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7853384, upper bound: 587.7837590
time: 13.74 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7869106, upper bound: 587.7855801
time: 12.67 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -331.6891174, 264.3709717, -319.2754822, 254.5992737, -586.2883911, 583.6464844
1: -279.8132935, 234.1618958, -269.4376526, 225.5053406, -505.3186340, 503.5995483
2: -366.4191589, 238.3235016, -352.8002625, 229.4174957, -595.8366089, 591.1237183
3: -387.8299255, 204.8932495, -373.2895813, 197.2380676, -585.0678711, 578.1828003
4: -357.2073364, 272.3690491, -343.9721985, 262.2610779, -619.4683838, 616.3412476
5: -319.2435913, 248.1821136, -307.3999329, 238.9307098, -558.1743164, 555.5820312
6: -305.4349670, 294.0406494, -294.0560608, 283.1358643, -588.5708008, 588.0966797
7: -333.2437439, 279.3079224, -320.7612305, 268.8778992, -602.1216431, 600.0691528
8: -402.3523560, 275.9558716, -387.3405762, 265.8261108, -668.1784668, 663.2964478
9: -303.0690918, 298.4181519, -291.8001099, 287.3647766, -590.4338379, 590.2182617

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7839541, upper bound: 587.7833882
time: 12.50 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7851724, upper bound: 587.7851677
time: 11.50 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -320.6063843, 255.6497498, -331.5371094, 264.3213196, -584.9277344, 587.1868286
1: -270.5463562, 226.4340210, -279.7626648, 234.1219177, -504.6682739, 506.1966553
2: -354.2686157, 230.3728485, -366.3517761, 238.2264252, -592.4949341, 596.7246094
3: -374.8429260, 198.0434723, -387.5408020, 204.6937103, -579.5364990, 585.5842285
4: -345.3756409, 263.3451538, -357.1849365, 272.3306885, -617.7062988, 620.5300903
5: -308.6768494, 239.9251251, -319.1701660, 248.1102448, -556.7871094, 559.0952148
6: -295.2798157, 284.3101196, -305.3431396, 293.9311829, -589.2109375, 589.6531372
7: -322.0983887, 269.9959717, -333.1170654, 279.1958923, -601.2942505, 603.1129761
8: -388.9247742, 266.8982544, -402.1542358, 275.9262085, -664.8509521, 669.0523682
9: -292.9961853, 288.5599060, -302.9768372, 298.3640442, -591.3602295, 591.5367432

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7853375, upper bound: 587.7837590
time: 12.28 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7869039, upper bound: 587.7855643
time: 12.98 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -332.2481689, 264.8117981, -330.1590576, 263.2323608, -595.4805298, 594.9707031
1: -280.2785339, 234.5524292, -278.6107483, 233.1509857, -513.4295044, 513.1631470
2: -367.0327148, 238.7207489, -364.8302307, 237.2484894, -604.2811890, 603.5509644
3: -388.4842224, 205.2321625, -385.9229431, 203.8515625, -592.3357544, 591.1550903
4: -357.8049622, 272.8241882, -355.6901855, 271.2076111, -629.0125732, 628.5144043
5: -319.7804565, 248.5966339, -317.8483582, 247.0890350, -566.8694458, 566.4450073
6: -305.9518738, 294.5336609, -304.0711975, 292.7108459, -598.6627197, 598.6048584
7: -333.8024902, 279.7765198, -331.7384033, 278.0406494, -611.8431396, 611.5148926
8: -403.0198059, 276.4082947, -400.4959412, 274.8006592, -677.8203735, 676.9042358
9: -303.5756836, 298.9157410, -301.7223206, 297.1340942, -600.7097168, 600.6380005

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7839500, upper bound: 587.7833882
time: 12.34 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7851642, upper bound: 587.7851642
time: 13.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.12 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7854149, upper bound: 587.7837974
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7870275, upper bound: 587.7856215
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7840322, upper bound: 587.7834056
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7852296, upper bound: 587.7851791
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7854149, upper bound: 587.7837974
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7870275, upper bound: 587.7856162
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7840310, upper bound: 587.7834056
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7840310, upper bound: 587.7851765
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7853384, upper bound: 587.7837590
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7869106, upper bound: 587.7855801
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7839541, upper bound: 587.7833882
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7851724, upper bound: 587.7851677
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7853375, upper bound: 587.7837590
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7869039, upper bound: 587.7855643
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7839500, upper bound: 587.7833882
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.12
Output dim: 6, lower bound: -587.7851642, upper bound: 587.7851642

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -309.7473450, 247.0285645, -320.7716370, 255.7764740, -565.5236816, 567.8001709
1: -261.3622131, 218.7798309, -270.6565247, 226.5557098, -487.9179077, 489.4363403
2: -342.2821960, 222.5552216, -354.4635315, 230.4990387, -572.7811279, 577.0186157
3: -362.1267395, 191.3647156, -374.9982605, 198.1660919, -560.2928467, 566.3629761
4: -333.6960754, 254.4400024, -345.5712891, 263.4924316, -597.1884766, 600.0112915
5: -298.2361450, 231.7957153, -308.8025208, 240.0469208, -538.2830811, 540.5981445
6: -285.2886047, 274.6909790, -295.4070435, 284.4291687, -569.7177124, 570.0980225
7: -311.1962585, 260.8607788, -322.2799072, 270.1466370, -581.3427124, 583.1406250
8: -375.7269897, 257.8581848, -389.1216431, 267.0430908, -642.7700195, 646.9798584
9: -283.0928650, 278.8180847, -293.1620789, 288.6941223, -571.7869873, 571.9801636

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7820941, upper bound: 587.7796351
time: 11.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7823195, upper bound: 587.7796550
time: 12.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -311.9342651, 248.7584686, -318.2188416, 253.7615051, -565.6957397, 566.9772949
1: -263.1957092, 220.3139496, -268.5485535, 224.7642365, -487.9598999, 488.8624268
2: -344.6782227, 224.0993347, -351.6492920, 228.6734772, -573.3516235, 575.7485962
3: -364.6857605, 192.6993713, -372.0472107, 196.5903015, -561.2760620, 564.7465820
4: -336.0426331, 256.2249146, -342.8467407, 261.3942566, -597.4367065, 599.0716553
5: -300.3410950, 233.4215088, -306.3782959, 238.1445618, -538.4856567, 539.7997437
6: -287.2969055, 276.6219788, -293.0824280, 282.2026062, -569.4995117, 569.7043457
7: -313.3763733, 262.6893921, -319.7096558, 267.9925842, -581.3689575, 582.3990479
8: -378.3565674, 259.6439514, -386.0672913, 264.9595337, -643.3160400, 645.7112427
9: -285.0795898, 280.7674255, -290.8390808, 286.4226990, -571.5021362, 571.6065063

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7845849, upper bound: 587.7828907
time: 13.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7846566, upper bound: 587.7829066
time: 13.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -321.2740784, 256.0977478, -319.4292603, 254.7160339, -575.9901123, 575.5269775
1: -270.9923706, 226.8140717, -269.5348816, 225.6100616, -496.6023560, 496.3489380
2: -354.9158020, 230.8267059, -352.9813843, 229.5459900, -584.4617920, 583.8081055
3: -375.6181030, 198.4838104, -373.4231873, 197.3462677, -572.9642944, 571.9069214
4: -345.9956665, 263.8231812, -344.1157227, 262.3988037, -608.3944702, 607.9389038
5: -309.2273254, 240.3752747, -307.5154724, 239.0520172, -548.2792969, 547.8906860
6: -295.8500671, 284.8117371, -294.1685791, 283.2409058, -579.0909424, 578.9803467
7: -322.7875061, 270.5436401, -320.9377136, 269.0214844, -591.8089600, 591.4812622
8: -389.6813660, 267.2794189, -387.5069885, 265.9470520, -655.6283569, 654.7862549
9: -293.5610046, 289.0631104, -291.9405823, 287.4962158, -581.0572510, 581.0036011

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7805367, upper bound: 587.7792182
time: 12.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7806101, upper bound: 587.7792055
time: 13.21 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -323.4786987, 257.8414307, -316.8480530, 252.6783447, -576.1570435, 574.6894531
1: -272.8403625, 228.3603516, -267.4028015, 223.7984009, -496.6387024, 495.7631531
2: -357.3312988, 232.3834686, -350.1356201, 227.7002869, -585.0314331, 582.5190430
3: -378.1976929, 199.8291626, -370.4378967, 195.7528839, -573.9505615, 570.2670898
4: -348.3608093, 265.6226501, -341.3601379, 260.2770386, -608.6378174, 606.9827881
5: -311.3488464, 242.0142212, -305.0633545, 237.1288300, -548.4776001, 547.0775757
6: -297.8744812, 286.7582092, -291.8175049, 280.9887085, -578.8631592, 578.5756226
7: -324.9851990, 272.3870850, -318.3384094, 266.8434143, -591.8284912, 590.7254639
8: -392.3321228, 269.0795898, -384.4178467, 263.8399658, -656.1721191, 653.4974365
9: -295.5639954, 291.0280762, -289.5913391, 285.1992188, -580.7631836, 580.6193237

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7825817, upper bound: 587.7824348
time: 11.28 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7824202, upper bound: 587.7824206
time: 12.51 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -310.3340149, 247.4917450, -333.3028259, 265.7161560, -576.0501709, 580.7945557
1: -261.8510742, 219.1900482, -281.2334900, 235.3566132, -497.2076721, 500.4235229
2: -342.9266663, 222.9725494, -368.3181458, 239.5262604, -582.4529419, 591.2907104
3: -362.8139648, 191.7212677, -389.5454712, 205.7918396, -568.6058350, 581.2667236
4: -334.3236694, 254.9181671, -359.0833435, 273.7940063, -608.1176758, 614.0014648
5: -298.7996826, 232.2313995, -320.8372803, 249.4416962, -548.2413330, 553.0686646
6: -285.8309937, 275.2089844, -306.9511108, 295.4638672, -581.2948608, 582.1600952
7: -311.7833557, 261.3534241, -334.9368286, 280.6960449, -592.4793701, 596.2902222
8: -376.4283447, 258.3337708, -404.2742920, 277.3855286, -653.8138428, 662.6080322
9: -283.6251831, 279.3414612, -304.5885620, 299.9560852, -583.5812988, 583.9300537

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7820941, upper bound: 587.7796335
time: 14.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7820926, upper bound: 587.7796539
time: 14.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -312.5036926, 249.2081451, -329.1060486, 262.3971863, -574.9008179, 578.3140869
1: -263.6703186, 220.7122345, -277.7243958, 232.4124756, -496.0827942, 498.4366455
2: -345.3040771, 224.5043793, -363.6828613, 236.5059204, -581.8099365, 588.1872559
3: -365.3532410, 193.0453949, -384.6848145, 203.2063141, -568.5595703, 577.7302246
4: -336.6518250, 256.6890564, -354.5692444, 270.3437500, -606.9956055, 611.2583008
5: -300.8882141, 233.8444519, -316.8299255, 246.3054657, -547.1936646, 550.6743774
6: -287.8233337, 277.1249390, -303.1007690, 291.7805481, -579.6038818, 580.2256470
7: -313.9463501, 263.1676636, -330.6906433, 277.1581421, -591.1044922, 593.8582764
8: -379.0375061, 260.1057129, -399.2264709, 273.9367065, -652.9741821, 659.3321533
9: -285.5963745, 281.2754211, -300.7651672, 296.1950989, -581.7914429, 582.0405884

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7820941, upper bound: 587.7828448
time: 14.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7846563, upper bound: 587.7828597
time: 13.20 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -321.8448181, 256.5478516, -331.9516602, 264.6485596, -586.4933472, 588.4993896
1: -271.4674072, 227.2127533, -280.1044312, 234.4048920, -505.8722534, 507.3171692
2: -355.5422058, 231.2324829, -366.8266602, 238.5672302, -594.1094360, 598.0590820
3: -376.2858887, 198.8300476, -387.9602356, 204.9665070, -581.2523804, 586.7902222
4: -346.6057129, 264.2879333, -357.6181641, 272.6931458, -619.2988281, 621.9061279
5: -309.7752991, 240.7986298, -319.5418701, 248.4401703, -558.2154541, 560.3403931
6: -296.3774109, 285.3149109, -305.7045898, 294.2679138, -590.6453247, 591.0195312
7: -323.3580627, 271.0222778, -333.5856934, 279.5635376, -602.9215698, 604.6079102
8: -390.3628845, 267.7413635, -402.6488953, 276.2819519, -666.6448364, 670.3902588
9: -294.0782471, 289.5715332, -303.3590698, 298.7502441, -592.8284912, 592.9305420

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7805367, upper bound: 587.7792182
time: 13.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7805184, upper bound: 587.7792055
time: 13.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -324.0334473, 258.2790527, -327.7301025, 261.3100586, -585.3435059, 586.0091553
1: -273.3021545, 228.7479706, -276.5744324, 231.4431305, -504.7453003, 505.3223267
2: -357.9402161, 232.7778931, -362.1638794, 235.5294647, -593.4696655, 594.9416504
3: -378.8469849, 200.1656647, -383.0697021, 202.3655396, -581.2125244, 583.2353516
4: -348.9539490, 266.0743713, -353.0771179, 269.2224731, -618.1763916, 619.1514282
5: -311.8815918, 242.4257660, -315.5102844, 245.2858124, -557.1673584, 557.9360352
6: -298.3870239, 287.2474365, -301.8308716, 290.5622864, -588.9493408, 589.0781860
7: -325.5398254, 272.8524170, -329.3142395, 276.0047607, -601.5445557, 602.1666260
8: -392.9946289, 269.5286255, -397.5709534, 272.8128662, -665.8074951, 667.0996094
9: -296.0668030, 291.5223083, -299.5127258, 294.9671936, -591.0339966, 591.0349731

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7825785, upper bound: 587.7824043
time: 13.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7824155, upper bound: 587.7823899
time: 11.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.43 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7820941, upper bound: 587.7796351
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7823195, upper bound: 587.7796550
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7845849, upper bound: 587.7828907
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7846566, upper bound: 587.7829066
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7805367, upper bound: 587.7792182
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7806101, upper bound: 587.7792055
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7825817, upper bound: 587.7824348
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7824202, upper bound: 587.7824206
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7820941, upper bound: 587.7796335
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7820926, upper bound: 587.7796539
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7820941, upper bound: 587.7828448
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7846563, upper bound: 587.7828597
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7805367, upper bound: 587.7792182
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7805184, upper bound: 587.7792055
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7825785, upper bound: 587.7824043
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.43
Output dim: 6, lower bound: -587.7824155, upper bound: 587.7823899
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.43
Output dim: 6, lower bound: -587.7853384, upper bound: 587.7837590
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.43
Output dim: 6, lower bound: -587.7869106, upper bound: 587.7855801
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.43
Output dim: 6, lower bound: -587.7839541, upper bound: 587.7833882
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.43
Output dim: 6, lower bound: -587.7851724, upper bound: 587.7851677
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.43
Output dim: 6, lower bound: -587.7853375, upper bound: 587.7837590
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.43
Output dim: 6, lower bound: -587.7869039, upper bound: 587.7855643
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.43
Output dim: 6, lower bound: -587.7839500, upper bound: 587.7833882
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.43
Output dim: 6, lower bound: -587.7851642, upper bound: 587.7851642
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=591.3155517578125
rel_dist={6: [-587.7906229930563, 587.7906229976039]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7902148, upper bound: 587.7902099
time: 14.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7901936, upper bound: 587.7901936
time: 17.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 31.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 31.92
Output dim: 6, lower bound: -587.7902148, upper bound: 587.7902099
IS_A2, status: Status.UNKNOWN, split count: 1, time: 31.92
Output dim: 6, lower bound: -587.7901936, upper bound: 587.7901936

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -316.2442627, 252.1643524, -323.1345520, 257.6471558, -573.8914185, 575.2988892
1: -266.8026123, 223.3411560, -272.6666260, 228.2141571, -495.0167847, 496.0077209
2: -349.4259644, 227.1557770, -357.0594177, 232.1644897, -581.5904541, 584.2152100
3: -369.7535400, 195.3359680, -377.8166504, 199.5932922, -569.3468018, 573.1525879
4: -340.6900635, 259.7447510, -348.1090698, 265.4132690, -606.1033325, 607.8536377
5: -304.4875183, 236.6209106, -311.1101990, 241.8050232, -546.2925415, 547.7310791
6: -291.2841187, 280.4365845, -297.6187439, 286.5498657, -577.8339233, 578.0552979
7: -317.6923523, 266.3090820, -324.6343384, 272.1219177, -589.8142700, 590.9432983
8: -383.5415039, 263.1564026, -391.9682007, 268.9574890, -652.4989624, 655.1246338
9: -289.0002136, 284.6189575, -295.3007507, 290.8226624, -579.8228760, 579.9196777

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7902077, upper bound: 587.7902045
time: 17.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7902077, upper bound: 587.7902045
time: 19.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -324.2376099, 258.5200195, -324.9851074, 259.1122131, -583.3496704, 583.5049438
1: -273.5869751, 228.9864655, -274.2134705, 229.5119019, -503.0988159, 503.1998901
2: -358.2707520, 232.9467773, -359.0911560, 233.4745331, -591.7453003, 592.0378418
3: -379.1147461, 200.2671814, -379.9938354, 200.7243958, -579.8391113, 580.2609863
4: -349.2968750, 266.3111877, -350.1053772, 266.9229431, -616.2198486, 616.4165649
5: -312.1709595, 242.6206055, -312.8906250, 243.1767578, -555.3477173, 555.5112305
6: -298.6396790, 287.5253296, -299.3288269, 288.1869812, -586.8265991, 586.8541260
7: -325.7356262, 273.0462036, -326.4826050, 273.6741028, -599.4096680, 599.5288086
8: -393.2973633, 269.8595886, -394.1965637, 270.4696655, -663.7670288, 664.0561523
9: -296.3012085, 291.8064270, -296.9840088, 292.4722290, -588.7734375, 588.7904053

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7901919, upper bound: 587.7901920
time: 19.86 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7901918, upper bound: 587.7901918
time: 19.15 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 40.10 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 40.10
Output dim: 6, lower bound: -587.7902077, upper bound: 587.7902045
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 40.10
Output dim: 6, lower bound: -587.7902077, upper bound: 587.7902045
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 40.10
Output dim: 6, lower bound: -587.7901919, upper bound: 587.7901920
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 40.10
Output dim: 6, lower bound: -587.7901918, upper bound: 587.7901918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -312.3135071, 249.0549927, -317.9436035, 253.5416412, -565.8551636, 566.9985962
1: -263.5118103, 220.5843048, -268.3223572, 224.5745697, -488.0863647, 488.9066772
2: -345.1010132, 224.3647308, -351.3483887, 228.4795227, -573.5805664, 575.7131348
3: -365.1347046, 192.9351807, -371.7194519, 196.4234161, -561.5581055, 564.6544800
4: -336.4733887, 256.5314331, -342.5410156, 261.1709290, -597.6442871, 599.0724487
5: -300.7032776, 233.6934052, -306.1136780, 237.9398956, -538.6431885, 539.8070679
6: -287.6549683, 276.9587402, -292.8267212, 281.9583435, -569.6132812, 569.7853394
7: -313.7533875, 263.0044861, -319.4331360, 267.7593384, -581.5126953, 582.4376221
8: -378.8215942, 259.9602051, -385.7369995, 264.7379456, -643.5594482, 645.6972046
9: -285.4293518, 281.1023560, -290.5859070, 286.1801758, -571.6094971, 571.6882324

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854888, upper bound: 587.7849878
time: 20.29 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7848233, upper bound: 587.7848108
time: 20.50 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -313.4273071, 249.9335785, -328.8609619, 262.2015381, -575.6287842, 578.7945557
1: -264.4393921, 221.3627625, -277.5235901, 232.2439423, -496.6833191, 498.8862915
2: -346.3242188, 225.1563721, -363.4152832, 236.3330688, -582.6572876, 588.5716553
3: -366.4389648, 193.6106873, -384.3927307, 203.0579681, -569.4969482, 578.0034180
4: -337.6642456, 257.4389954, -354.2963562, 270.1452026, -607.8093872, 611.7353516
5: -301.7727356, 234.5203094, -316.5943604, 246.1231995, -547.8958740, 551.1145020
6: -288.6843262, 277.9413757, -302.8726196, 291.5632019, -580.2475586, 580.8139648
7: -314.8678284, 263.9393311, -330.4443054, 276.9502258, -591.8179321, 594.3836670
8: -380.1520081, 260.8616333, -398.9326172, 273.7400208, -653.8920288, 659.7942505
9: -286.4398193, 282.0952454, -300.5396729, 295.9798279, -582.4196167, 582.6348877

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854888, upper bound: 587.7849874
time: 20.14 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7848221, upper bound: 587.7848101
time: 19.11 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -320.3144531, 255.4163513, -319.7540894, 254.9749908, -575.2894287, 575.1704102
1: -270.3024292, 226.2348328, -269.8355408, 225.8442535, -496.1466675, 496.0703430
2: -353.9535828, 230.1613007, -353.3360596, 229.7610321, -583.7145996, 583.4973755
3: -374.5051270, 197.8704834, -373.8495789, 197.5299377, -572.0349731, 571.7200928
4: -345.0877075, 263.1040039, -344.4941101, 262.6477051, -607.7354126, 607.5981445
5: -308.3936768, 239.6990814, -307.8555908, 239.2819824, -547.6756592, 547.5546875
6: -295.0169067, 284.0537720, -294.4994507, 283.5599670, -578.5769043, 578.5532227
7: -321.8042603, 269.7480469, -321.2412415, 269.2777405, -591.0820312, 590.9892578
8: -388.5862122, 266.6690674, -387.9170837, 266.2173767, -654.8034668, 654.5861816
9: -292.7369995, 288.2966919, -292.2326355, 287.7941895, -580.5311279, 580.5292969

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854503, upper bound: 587.7849726
time: 21.39 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7848034, upper bound: 587.7848032
time: 16.83 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -321.4246826, 256.2917786, -330.6366882, 263.6075745, -585.0322266, 586.9284058
1: -271.2267456, 227.0105591, -279.0079041, 233.4892883, -504.7160339, 506.0184326
2: -355.1724548, 230.9501801, -365.3650208, 237.5912933, -592.7637329, 596.3151245
3: -375.8045044, 198.5433655, -386.4822693, 204.1430206, -579.9475098, 585.0255737
4: -346.2747803, 264.0083313, -356.2113953, 271.5935364, -617.8682251, 620.2197266
5: -309.4600220, 240.5228729, -318.3033142, 247.4397888, -556.8997803, 558.8261719
6: -296.0432739, 285.0330200, -304.5141296, 293.1344910, -589.1777344, 589.5471191
7: -322.9144287, 270.6791687, -332.2176514, 278.4396973, -601.3541260, 602.8968506
8: -389.9117126, 267.5670471, -401.0719299, 275.1913757, -665.1030884, 668.6387939
9: -293.7437134, 289.2851562, -302.1540527, 297.5626831, -591.3063965, 591.4392090

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854480, upper bound: 587.7849685
time: 19.19 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7848019, upper bound: 587.7848019
time: 15.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 35.67 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 35.67
Output dim: 6, lower bound: -587.7854888, upper bound: 587.7849878
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 35.67
Output dim: 6, lower bound: -587.7848233, upper bound: 587.7848108
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 35.67
Output dim: 6, lower bound: -587.7854888, upper bound: 587.7849874
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 35.67
Output dim: 6, lower bound: -587.7848221, upper bound: 587.7848101
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 35.67
Output dim: 6, lower bound: -587.7854503, upper bound: 587.7849726
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 35.67
Output dim: 6, lower bound: -587.7848034, upper bound: 587.7848032
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 35.67
Output dim: 6, lower bound: -587.7854480, upper bound: 587.7849685
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 35.67
Output dim: 6, lower bound: -587.7848019, upper bound: 587.7848019

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -310.1713257, 247.3647461, -316.5679016, 252.4561157, -562.6274414, 563.9325562
1: -261.7220459, 219.0790863, -267.1729126, 223.6077118, -485.3297424, 486.2519836
2: -342.7400818, 222.8475800, -349.8320923, 227.5052338, -570.2451782, 572.6796875
3: -362.6160278, 191.6232758, -370.1021118, 195.5808868, -558.1968994, 561.7253418
4: -334.1524963, 254.7859650, -341.0504761, 260.0501709, -594.2026367, 595.8364258
5: -298.6455383, 232.1091309, -304.7922058, 236.9224548, -535.5679932, 536.9013672
6: -285.6707153, 275.0640259, -291.5524597, 280.7414551, -566.4121704, 566.6164551
7: -311.6102600, 261.2089233, -318.0568237, 266.6062317, -578.2164917, 579.2657471
8: -376.2422485, 258.2119446, -384.0805359, 263.6152039, -639.8574219, 642.2924805
9: -283.4797668, 279.1918640, -289.3338623, 284.9531860, -568.4329224, 568.5257568

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7837483, upper bound: 587.7831278
time: 18.89 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854888, upper bound: 587.7849878
time: 17.39 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -321.4164429, 256.1982117, -315.0408936, 251.2486725, -572.6651001, 571.2390747
1: -271.1030273, 226.9076538, -265.8966064, 222.5304108, -493.6334229, 492.8042603
2: -355.0527954, 230.9140472, -348.1442566, 226.4194183, -581.4722290, 579.0582275
3: -375.7538757, 198.5514679, -368.3112183, 194.6496277, -570.4035034, 566.8625488
4: -346.1384888, 263.9347839, -339.3938293, 258.8052979, -604.9437866, 603.3286133
5: -309.3564453, 240.4717865, -303.3289185, 235.7897949, -545.1461792, 543.8006592
6: -295.9667358, 284.9177856, -290.1471863, 279.3883667, -575.3551025, 575.0649414
7: -322.9149780, 270.6425476, -316.5297241, 265.3261108, -588.2410889, 587.1722412
8: -389.8343811, 267.3813477, -382.2442627, 262.3670044, -652.2012329, 649.6256104
9: -293.6853333, 289.1614685, -287.9440918, 283.5898132, -577.2751465, 577.1055908

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7832757, upper bound: 587.7829821
time: 20.55 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7848233, upper bound: 587.7848108
time: 16.16 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -311.2841797, 248.2425232, -327.4866943, 261.1170959, -572.4012451, 575.7292480
1: -262.6487427, 219.8567657, -276.3753357, 231.2782593, -493.9270020, 496.2320862
2: -343.9620667, 223.6386719, -361.9006042, 235.3599091, -579.3219604, 585.5393066
3: -363.9190674, 192.2983246, -382.7773438, 202.2163391, -566.1353760, 575.0756836
4: -335.3423462, 255.6926117, -352.8073425, 269.0256653, -604.3680420, 608.4998779
5: -299.7140503, 232.9351501, -315.2743530, 245.1066284, -544.8206177, 548.2094116
6: -286.6991272, 276.0457764, -301.5996399, 290.3477783, -577.0468750, 577.6453247
7: -312.7237549, 262.1428833, -329.0695190, 275.7981262, -588.5218506, 591.2122803
8: -377.5716248, 259.1125793, -397.2779236, 272.6184692, -650.1900635, 656.3905029
9: -284.4892273, 280.1839294, -299.2889099, 294.7540588, -579.2432251, 579.4728394

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7837483, upper bound: 587.7831278
time: 20.73 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854888, upper bound: 587.7849874
time: 20.01 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -322.5117798, 257.0618591, -325.9488525, 259.9009399, -582.4127197, 583.0106201
1: -272.0145264, 227.6727295, -275.0898132, 230.1932831, -502.2077942, 502.7625427
2: -356.2551575, 231.6925812, -360.2009277, 234.2667694, -590.5219116, 591.8934937
3: -377.0355225, 199.2153931, -380.9738464, 201.2779541, -578.3134766, 580.1892090
4: -347.3093567, 264.8267212, -351.1383362, 267.7717285, -615.0810547, 615.9650879
5: -310.4080505, 241.2842560, -313.8005981, 243.9656830, -554.3737183, 555.0848389
6: -296.9785156, 285.8834839, -300.1837769, 288.9847107, -585.9631958, 586.0671997
7: -324.0101929, 271.5610046, -327.5312195, 274.5089417, -598.5191650, 599.0922241
8: -391.1419678, 268.2674866, -395.4282532, 271.3609924, -662.5028687, 663.6956787
9: -294.6781006, 290.1370850, -297.8888550, 293.3808594, -588.0589600, 588.0259399

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7832757, upper bound: 587.7829821
time: 19.03 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7848221, upper bound: 587.7848101
time: 14.94 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -318.1926880, 253.7420197, -318.3839722, 253.8938293, -572.0865479, 572.1259766
1: -268.5295105, 224.7436523, -268.6906433, 224.8813324, -493.4108276, 493.4342041
2: -351.6150208, 228.6586456, -351.8259583, 228.7907410, -580.4057617, 580.4846191
3: -372.0106201, 196.5712891, -372.2387695, 196.6910248, -568.7015991, 568.8100586
4: -342.7886963, 261.3753052, -343.0095215, 261.5315247, -604.3201904, 604.3848267
5: -306.3555298, 238.1297150, -306.5395203, 238.2686768, -544.6242065, 544.6692505
6: -293.0517273, 282.1769104, -293.2304688, 282.3481140, -575.3998413, 575.4073486
7: -319.6814880, 267.9693909, -319.8705750, 268.1292114, -587.8106689, 587.8399048
8: -386.0313721, 264.9376526, -386.2673645, 265.0995178, -651.1307373, 651.2050171
9: -290.8056335, 286.4043274, -290.9855347, 286.5721130, -577.3777466, 577.3898926

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7837044, upper bound: 587.7831149
time: 22.83 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7854503, upper bound: 587.7849726
time: 19.51 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -329.6528015, 262.7521057, -316.8469543, 252.6784058, -582.3311768, 579.5990601
1: -278.1025391, 232.7274780, -267.4059143, 223.7967377, -501.8992310, 500.1333923
2: -364.1722717, 236.8729858, -350.1268616, 227.6977997, -591.8699951, 586.9997559
3: -385.4247742, 203.6348877, -370.4361877, 195.7532959, -581.1781006, 574.0708618
4: -355.0160828, 270.7023621, -341.3420410, 260.2786865, -615.2947998, 612.0442505
5: -317.2781067, 246.6613312, -305.0666199, 237.1284943, -554.4065552, 551.7279663
6: -303.5541077, 292.2283325, -291.8158264, 280.9859924, -584.5401001, 584.0441895
7: -331.2001648, 277.5911560, -318.3333130, 266.8406677, -598.0408325, 595.9243774
8: -399.8959045, 274.2824402, -384.4187927, 263.8431091, -663.7390137, 658.7012329
9: -301.2157593, 296.5887146, -289.5867004, 285.1996460, -586.4152832, 586.1753540

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7832476, upper bound: 587.7829756
time: 22.87 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7848034, upper bound: 587.7848032
time: 16.00 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -319.3009338, 254.6158142, -329.2694397, 262.5285950, -581.8295288, 583.8851318
1: -269.4520874, 225.5178375, -277.8655090, 232.5284271, -501.9804688, 503.3833008
2: -352.8316040, 229.4461823, -363.8580322, 236.6230469, -589.4546509, 593.3041992
3: -373.3075562, 197.2431030, -384.8751221, 203.3058472, -576.6134033, 582.1182251
4: -343.9735413, 262.2779541, -354.7297974, 270.4797668, -614.4533081, 617.0077515
5: -307.4198608, 238.9519958, -316.9899902, 246.4282532, -553.8481445, 555.9418945
6: -294.0762329, 283.1544800, -303.2476501, 291.9252319, -586.0014038, 586.4020996
7: -320.7897034, 268.8987122, -330.8498535, 277.2934570, -598.0831299, 599.7485352
8: -387.3546753, 265.8340759, -399.4257812, 274.0757141, -661.4303589, 665.2598877
9: -291.8105164, 287.3908691, -300.9095154, 296.3432312, -588.1536865, 588.3004150

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7837044, upper bound: 587.7831149
time: 20.32 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7837044, upper bound: 587.7849685
time: 18.75 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -330.7567749, 263.6221924, -327.7213440, 261.3044128, -592.0610352, 591.3433838
1: -279.0209045, 233.4982910, -276.5712585, 231.4360962, -510.4569702, 510.0695190
2: -365.3836365, 237.6573181, -362.1468506, 235.5227356, -600.9063721, 599.8040771
3: -386.7161865, 204.3036652, -383.0592651, 202.3609467, -589.0771484, 587.3629150
4: -356.1958618, 271.6010437, -353.0496826, 269.2174988, -625.4133301, 624.6507568
5: -318.3380737, 247.4797974, -315.5065308, 245.2797699, -563.6177979, 562.9863281
6: -304.5743408, 293.2014465, -301.8223267, 290.5530396, -595.1273193, 595.0237427
7: -332.3034668, 278.5161438, -329.3012390, 275.9956970, -608.2991333, 607.8172607
8: -401.2132874, 275.1750488, -397.5632935, 272.8097839, -674.0230713, 672.7383423
9: -302.2159119, 297.5708008, -299.5000610, 294.9608154, -597.1767578, 597.0708618

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7832476, upper bound: 587.7829756
time: 17.99 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -587.7848019, upper bound: 587.7848019
time: 18.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 37.37 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7837483, upper bound: 587.7831278
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7854888, upper bound: 587.7849878
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7832757, upper bound: 587.7829821
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7848233, upper bound: 587.7848108
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7837483, upper bound: 587.7831278
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7854888, upper bound: 587.7849874
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7832757, upper bound: 587.7829821
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7848221, upper bound: 587.7848101
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7837044, upper bound: 587.7831149
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7854503, upper bound: 587.7849726
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7832476, upper bound: 587.7829756
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7848034, upper bound: 587.7848032
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7837044, upper bound: 587.7831149
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7837044, upper bound: 587.7849685
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7832476, upper bound: 587.7829756
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.37
Output dim: 6, lower bound: -587.7848019, upper bound: 587.7848019
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=591.3155517578125
rel_dist={6: [-587.7904223681265, 587.7904223711924]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1824.74 seconds
