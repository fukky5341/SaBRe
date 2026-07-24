## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 202.485480649
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-130.5882568, 104.3158493, -130.5882568, 104.3158493, -234.9041138, 234.9041138)
1: (-111.0153198, 92.5080414, -111.0153198, 92.5080414, -203.5233612, 203.5233612)
2: (-144.5845337, 93.9274826, -144.5845337, 93.9274826, -238.5120239, 238.5120239)
3: (-152.5511169, 81.2579956, -152.5511169, 81.2579956, -233.8091125, 233.8091125)
4: (-140.5548248, 108.2794342, -140.5548248, 108.2794342, -248.8342590, 248.8342590)
5: (-125.6117477, 97.8683395, -125.6117477, 97.8683395, -223.4800873, 223.4800873)
6: (-120.6793823, 116.6907272, -120.6793823, 116.6907272, -237.3701172, 237.3701172)
7: (-131.0779724, 110.4361649, -131.0779724, 110.4361649, -241.5141144, 241.5141144)
8: (-159.4465179, 110.3260040, -159.4465179, 110.3260040, -269.7725220, 269.7725220)
9: (-119.6686935, 118.7806778, -119.6686935, 118.7806778, -238.4493713, 238.4493713)

## BASE Result
execution time: IAR + LP analysis = 1.09 + 9.77 = 10.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -202.6091898, upper bound: 202.6091898


# Binary Search by BASE starts (time budget: 2689.14 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=203.5233612060547
rel_dist={1: [-202.60902678108835, 202.60902678108835]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=203.5233612060547
rel_dist={1: [-202.60871310808878, 202.60871310808875]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=203.5233612060547
rel_dist={1: [-202.608202498089, 202.60820249808899]}

## Binary Search Result
Binary search time: 37.81 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2651.34 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6052398, upper bound: 202.6050528
time: 7.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6049940, upper bound: 202.6049940
time: 8.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.84
Output dim: 1, lower bound: -202.6052398, upper bound: 202.6050528
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.84
Output dim: 1, lower bound: -202.6049940, upper bound: 202.6049940

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -123.0395737, 98.3348694, -130.4325256, 104.1927185, -227.2322693, 228.7673950
1: -104.6226501, 87.1827240, -110.8831253, 92.3981705, -197.0208130, 198.0658569
2: -136.2332764, 88.5248413, -144.4122314, 93.8163376, -230.0495911, 232.9370575
3: -143.6921082, 76.5960464, -152.3682404, 81.1613464, -224.8534546, 228.9642944
4: -132.4142151, 102.0421982, -140.3868561, 108.1505661, -240.5647888, 242.4290466
5: -118.3712463, 92.2120056, -125.4628830, 97.7515106, -216.1227264, 217.6748962
6: -113.7487183, 109.9841461, -120.5365372, 116.5522385, -230.3009644, 230.5206909
7: -123.4394455, 104.0390549, -130.9197998, 110.3039932, -233.7434082, 234.9588318
8: -150.3356781, 104.1154709, -159.2592010, 110.1981659, -260.5338135, 263.3745728
9: -112.7398911, 111.9838409, -119.5254288, 118.6399841, -231.3798828, 231.5092468

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6049747, upper bound: 202.6049747
time: 7.84 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6049747, upper bound: 202.6049750
time: 6.77 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.7521286, 100.5080948, -130.5731812, 104.3039780, -230.0561066, 231.0812531
1: -106.9180679, 89.1028976, -111.0025330, 92.4974136, -199.4154663, 200.1054382
2: -139.2514191, 90.4931870, -144.5678864, 93.9167633, -233.1681824, 235.0610657
3: -146.8589630, 78.2464371, -152.5334015, 81.2485962, -228.1075592, 230.7798309
4: -135.3592224, 104.2918320, -140.5385895, 108.2669983, -243.6261749, 244.8304138
5: -120.9924850, 94.2561188, -125.5973206, 97.8570786, -218.8495331, 219.8534393
6: -116.2524948, 112.4022141, -120.6655502, 116.6773529, -232.9298401, 233.0677643
7: -126.1867523, 106.3574982, -131.0627594, 110.4234314, -236.6101837, 237.4202576
8: -153.6653137, 106.3638687, -159.4284515, 110.3136597, -263.9788818, 265.7923279
9: -115.2547913, 114.4284286, -119.6549301, 118.7671051, -234.0218964, 234.0833588

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6049750, upper bound: 202.6049879
time: 6.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6049750, upper bound: 202.6049940
time: 6.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.44 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.44
Output dim: 1, lower bound: -202.6049747, upper bound: 202.6049747
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.44
Output dim: 1, lower bound: -202.6049747, upper bound: 202.6049750
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.44
Output dim: 1, lower bound: -202.6049750, upper bound: 202.6049879
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.44
Output dim: 1, lower bound: -202.6049750, upper bound: 202.6049940

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -123.0395737, 98.3348694, -123.0395737, 98.3348694, -221.3744202, 221.3744202
1: -104.6226501, 87.1827240, -104.6226501, 87.1827240, -191.8053741, 191.8053741
2: -136.2332764, 88.5248413, -136.2332764, 88.5248413, -224.7581177, 224.7581177
3: -143.6921082, 76.5960464, -143.6921082, 76.5960464, -220.2881470, 220.2881470
4: -132.4142151, 102.0421982, -132.4142151, 102.0421982, -234.4564209, 234.4564056
5: -118.3712463, 92.2120056, -118.3712463, 92.2120056, -210.5832520, 210.5832520
6: -113.7487183, 109.9841461, -113.7487183, 109.9841461, -223.7328644, 223.7328644
7: -123.4394455, 104.0390549, -123.4394455, 104.0390549, -227.4784851, 227.4784851
8: -150.3356781, 104.1154709, -150.3356781, 104.1154709, -254.4511414, 254.4511414
9: -112.7398911, 111.9838409, -112.7398911, 111.9838409, -224.7237244, 224.7237244

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5997551, upper bound: 202.5995704
time: 7.06 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6000154, upper bound: 202.5998714
time: 8.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -123.0395737, 98.3348694, -125.7521286, 100.5080948, -223.5476532, 224.0870056
1: -104.6226501, 87.1827240, -106.9180679, 89.1028976, -193.7255554, 194.1007538
2: -136.2332764, 88.5248413, -139.2514191, 90.4931870, -226.7264709, 227.7762451
3: -143.6921082, 76.5960464, -146.8589630, 78.2464371, -221.9385376, 223.4550171
4: -132.4142151, 102.0421982, -135.3592224, 104.2918320, -236.7060394, 237.4014130
5: -118.3712463, 92.2120056, -120.9924850, 94.2561188, -212.6273346, 213.2044983
6: -113.7487183, 109.9841461, -116.2524948, 112.4022141, -226.1509399, 226.2366333
7: -123.4394455, 104.0390549, -126.1867523, 106.3574982, -229.7969360, 230.2257996
8: -150.3356781, 104.1154709, -153.6653137, 106.3638687, -256.6995544, 257.7807617
9: -112.7398911, 111.9838409, -115.2547913, 114.4284286, -227.1683197, 227.2386322

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5997551, upper bound: 202.5995704
time: 6.51 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6000154, upper bound: 202.5998714
time: 7.50 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -125.7521286, 100.5080948, -123.0395737, 98.3348694, -224.0870056, 223.5476532
1: -106.9180679, 89.1028976, -104.6226501, 87.1827240, -194.1007538, 193.7255554
2: -139.2514191, 90.4931870, -136.2332764, 88.5248413, -227.7762451, 226.7264709
3: -146.8589630, 78.2464371, -143.6921082, 76.5960464, -223.4550171, 221.9385376
4: -135.3592224, 104.2918320, -132.4142151, 102.0421982, -237.4013977, 236.7060394
5: -120.9924850, 94.2561188, -118.3712463, 92.2120056, -213.2044983, 212.6273346
6: -116.2524948, 112.4022141, -113.7487183, 109.9841461, -226.2366333, 226.1509399
7: -126.1867523, 106.3574982, -123.4394455, 104.0390549, -230.2257996, 229.7969360
8: -153.6653137, 106.3638687, -150.3356781, 104.1154709, -257.7807617, 256.6995239
9: -115.2547913, 114.4284286, -112.7398911, 111.9838409, -227.2386169, 227.1683197

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5994021, upper bound: 202.5994651
time: 7.37 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5998490, upper bound: 202.5998744
time: 7.08 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -125.7521286, 100.5080948, -125.7521286, 100.5080948, -226.2602234, 226.2602234
1: -106.9180679, 89.1028976, -106.9180679, 89.1028976, -196.0209503, 196.0209503
2: -139.2514191, 90.4931870, -139.2514191, 90.4931870, -229.7445984, 229.7445984
3: -146.8589630, 78.2464371, -146.8589630, 78.2464371, -225.1054077, 225.1054077
4: -135.3592224, 104.2918320, -135.3592224, 104.2918320, -239.6510315, 239.6510315
5: -120.9924850, 94.2561188, -120.9924850, 94.2561188, -215.2485962, 215.2485962
6: -116.2524948, 112.4022141, -116.2524948, 112.4022141, -228.6546936, 228.6546936
7: -126.1867523, 106.3574982, -126.1867523, 106.3574982, -232.5442505, 232.5442505
8: -153.6653137, 106.3638687, -153.6653137, 106.3638687, -260.0291443, 260.0291443
9: -115.2547913, 114.4284286, -115.2547913, 114.4284286, -229.6832275, 229.6832275

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5994021, upper bound: 202.5994673
time: 8.11 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5998490, upper bound: 202.5998882
time: 7.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.19
Output dim: 1, lower bound: -202.5997551, upper bound: 202.5995704
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.19
Output dim: 1, lower bound: -202.6000154, upper bound: 202.5998714
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.19
Output dim: 1, lower bound: -202.5997551, upper bound: 202.5995704
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.19
Output dim: 1, lower bound: -202.6000154, upper bound: 202.5998714
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.19
Output dim: 1, lower bound: -202.5994021, upper bound: 202.5994651
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.19
Output dim: 1, lower bound: -202.5998490, upper bound: 202.5998744
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.19
Output dim: 1, lower bound: -202.5994021, upper bound: 202.5994673
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.19
Output dim: 1, lower bound: -202.5998490, upper bound: 202.5998882

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -114.3123093, 91.4359589, -123.0395737, 98.3348694, -212.6471863, 214.4755249
1: -97.3022995, 81.0540771, -104.6226501, 87.1827240, -184.4850159, 185.6767273
2: -126.5931625, 82.2796860, -136.2332764, 88.5248413, -215.1179962, 218.5129547
3: -133.3694153, 71.1658020, -143.6921082, 76.5960464, -209.9654541, 214.8579102
4: -123.0269012, 94.8894196, -132.4142151, 102.0421982, -225.0690918, 227.3036194
5: -109.9164963, 85.7271194, -118.3712463, 92.2120056, -202.1285095, 204.0982971
6: -105.6988068, 102.2890015, -113.7487183, 109.9841461, -215.6829529, 216.0377197
7: -114.6357803, 96.7082825, -123.4394455, 104.0390549, -218.6748047, 220.1477203
8: -139.8174286, 96.9648514, -150.3356781, 104.1154709, -243.9328918, 247.3005219
9: -104.8073578, 104.1193542, -112.7398911, 111.9838409, -216.7911682, 216.8592529

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5997421, upper bound: 202.5997421
time: 7.25 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5997421, upper bound: 202.5998656
time: 7.54 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -117.8387451, 94.2534485, -122.4439087, 97.8646011, -215.7033386, 216.6973419
1: -100.2893982, 83.5374603, -104.1230927, 86.7641602, -187.0535583, 187.6605072
2: -130.5082703, 84.7994843, -135.5762177, 88.0994949, -218.6077423, 220.3756714
3: -137.5389099, 73.3317947, -142.9889221, 76.2262115, -213.7651215, 216.3207092
4: -126.8203888, 97.7702942, -131.7720795, 101.5527420, -228.3731384, 229.5423584
5: -113.3322678, 88.3442764, -117.7955551, 91.7691498, -205.1014099, 206.1398315
6: -108.9602051, 105.4201050, -113.1996536, 109.4584122, -218.4185791, 218.6197510
7: -118.1710663, 99.6975708, -122.8380661, 103.5399857, -221.7110596, 222.5356445
8: -144.1304474, 99.8603821, -149.6181183, 103.6270523, -247.7574921, 249.4785004
9: -108.0024261, 107.2640152, -112.1967926, 111.4449463, -219.4473724, 219.4608154

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5962747, upper bound: 202.5963576
time: 6.86 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5966171, upper bound: 202.5966171
time: 7.09 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -114.3123093, 91.4359589, -125.7521286, 100.5080948, -214.8204041, 217.1880798
1: -97.3022995, 81.0540771, -106.9180679, 89.1028976, -186.4051971, 187.9721375
2: -126.5931625, 82.2796860, -139.2514191, 90.4931870, -217.0863495, 221.5310974
3: -133.3694153, 71.1658020, -146.8589630, 78.2464371, -211.6158447, 218.0247650
4: -123.0269012, 94.8894196, -135.3592224, 104.2918320, -227.3187256, 230.2486115
5: -109.9164963, 85.7271194, -120.9924850, 94.2561188, -204.1725769, 206.7195587
6: -105.6988068, 102.2890015, -116.2524948, 112.4022141, -218.1010132, 218.5414886
7: -114.6357803, 96.7082825, -126.1867523, 106.3574982, -220.9932709, 222.8950195
8: -139.8174286, 96.9648514, -153.6653137, 106.3638687, -246.1813049, 250.6301575
9: -104.8073578, 104.1193542, -115.2547913, 114.4284286, -219.2357788, 219.3741455

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5994857, upper bound: 202.5992637
time: 7.94 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5994857, upper bound: 202.5995501
time: 8.09 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -117.8387451, 94.2534485, -125.1533661, 100.0352936, -217.8740082, 219.4068146
1: -100.2893982, 83.5374603, -106.4159088, 88.6821518, -188.9715576, 189.9533386
2: -130.5082703, 84.7994843, -138.5910187, 90.0656967, -220.5739746, 223.3904877
3: -137.5389099, 73.3317947, -146.1517792, 77.8745804, -215.4134674, 219.4835815
4: -126.8203888, 97.7702942, -134.7138519, 103.7997360, -230.6201172, 232.4841309
5: -113.3322678, 88.3442764, -120.4135666, 93.8111572, -207.1434326, 208.7578430
6: -108.9602051, 105.4201050, -115.7005005, 111.8737335, -220.8338928, 221.1206055
7: -118.1710663, 99.6975708, -125.5824509, 105.8559875, -224.0270386, 225.2800140
8: -144.1304474, 99.8603821, -152.9439850, 105.8729172, -250.0033417, 252.8043671
9: -108.0024261, 107.2640152, -114.7087631, 113.8868484, -221.8892670, 221.9727783

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5961115, upper bound: 202.5960013
time: 7.99 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5965362, upper bound: 202.5964392
time: 7.60 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -116.9979630, 93.5887604, -123.0395737, 98.3348694, -215.3328247, 216.6282806
1: -99.5721359, 82.9556580, -104.6226501, 87.1827240, -186.7548370, 187.5783081
2: -129.5791321, 84.2278137, -136.2332764, 88.5248413, -218.1039734, 220.4610748
3: -136.5052948, 72.8001862, -143.6921082, 76.5960464, -213.1013489, 216.4922791
4: -125.9418793, 97.1169586, -132.4142151, 102.0421982, -227.9840698, 229.5311737
5: -112.5123062, 87.7526932, -118.3712463, 92.2120056, -204.7243042, 206.1239014
6: -108.1777878, 104.6804886, -113.7487183, 109.9841461, -218.1619263, 218.4291992
7: -117.3524323, 99.0018997, -123.4394455, 104.0390549, -221.3914795, 222.4413300
8: -143.1133118, 99.1928787, -150.3356781, 104.1154709, -247.2287903, 249.5285645
9: -107.2974091, 106.5389862, -112.7398911, 111.9838409, -219.2812347, 219.2788696

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5992637, upper bound: 202.5994857
time: 7.48 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5992637, upper bound: 202.5996185
time: 7.27 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -120.5904770, 96.4575043, -122.4439087, 97.8646011, -218.4550781, 218.9013519
1: -102.6173248, 85.4852600, -104.1230927, 86.7641602, -189.3814697, 189.6083069
2: -133.5718536, 86.7935028, -135.5762177, 88.0994949, -221.6713104, 222.3697205
3: -140.7533112, 75.0046387, -142.9889221, 76.2262115, -216.9795227, 217.9935608
4: -129.8094025, 100.0519791, -131.7720795, 101.5527420, -231.3621521, 231.8240662
5: -115.9898376, 90.4180756, -117.7955551, 91.7691498, -207.7589874, 208.2136230
6: -111.4987869, 107.8731232, -113.1996536, 109.4584122, -220.9571838, 221.0727844
7: -120.9549789, 102.0482559, -122.8380661, 103.5399857, -224.4949493, 224.8863220
8: -147.5045624, 102.1433716, -149.6181183, 103.6270523, -251.1316223, 251.7614746
9: -110.5523911, 109.7455215, -112.1967926, 111.4449463, -221.9973145, 221.9423065

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5960256, upper bound: 202.5962403
time: 6.95 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5964392, upper bound: 202.5965362
time: 7.67 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -116.9979630, 93.5887604, -125.7521286, 100.5080948, -217.5060577, 219.3408813
1: -99.5721359, 82.9556580, -106.9180679, 89.1028976, -188.6750336, 189.8737183
2: -129.5791321, 84.2278137, -139.2514191, 90.4931870, -220.0723267, 223.4792175
3: -136.5052948, 72.8001862, -146.8589630, 78.2464371, -214.7517395, 219.6591492
4: -125.9418793, 97.1169586, -135.3592224, 104.2918320, -230.2337036, 232.4761810
5: -112.5123062, 87.7526932, -120.9924850, 94.2561188, -206.7684174, 208.7451630
6: -108.1777878, 104.6804886, -116.2524948, 112.4022141, -220.5800018, 220.9329529
7: -117.3524323, 99.0018997, -126.1867523, 106.3574982, -223.7099304, 225.1886292
8: -143.1133118, 99.1928787, -153.6653137, 106.3638687, -249.4771729, 252.8581848
9: -107.2974091, 106.5389862, -115.2547913, 114.4284286, -221.7258301, 221.7937775

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5991963, upper bound: 202.5992060
time: 7.45 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5991963, upper bound: 202.5994604
time: 6.96 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -120.5904770, 96.4575043, -125.1533661, 100.0352936, -220.6257629, 221.6108398
1: -102.6173248, 85.4852600, -106.4159088, 88.6821518, -191.2994537, 191.9011383
2: -133.5718536, 86.7935028, -138.5910187, 90.0656967, -223.6375427, 225.3845215
3: -140.7533112, 75.0046387, -146.1517792, 77.8745804, -218.6278839, 221.1564026
4: -129.8094025, 100.0519791, -134.7138519, 103.7997360, -233.6091156, 234.7658386
5: -115.9898376, 90.4180756, -120.4135666, 93.8111572, -209.8009949, 210.8316345
6: -111.4987869, 107.8731232, -115.7005005, 111.8737335, -223.3724823, 223.5736237
7: -120.9549789, 102.0482559, -125.5824509, 105.8559875, -226.8109283, 227.6307068
8: -147.5045624, 102.1433716, -152.9439850, 105.8729172, -253.3774414, 255.0873413
9: -110.5523911, 109.7455215, -114.7087631, 113.8868484, -224.4392090, 224.4542694

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5960419, upper bound: 202.5960668
time: 8.31 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5965057, upper bound: 202.5965129
time: 7.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.52 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5997421, upper bound: 202.5997421
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5997421, upper bound: 202.5998656
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5962747, upper bound: 202.5963576
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5966171, upper bound: 202.5966171
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5994857, upper bound: 202.5992637
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5994857, upper bound: 202.5995501
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5961115, upper bound: 202.5960013
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5965362, upper bound: 202.5964392
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5992637, upper bound: 202.5994857
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5992637, upper bound: 202.5996185
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5960256, upper bound: 202.5962403
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5964392, upper bound: 202.5965362
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5991963, upper bound: 202.5992060
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5991963, upper bound: 202.5994604
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5960419, upper bound: 202.5960668
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.52
Output dim: 1, lower bound: -202.5965057, upper bound: 202.5965129

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -114.3123093, 91.4359589, -114.3123093, 91.4359589, -205.7482605, 205.7482605
1: -97.3022995, 81.0540771, -97.3022995, 81.0540771, -178.3563843, 178.3563843
2: -126.5931625, 82.2796860, -126.5931625, 82.2796860, -208.8728485, 208.8728485
3: -133.3694153, 71.1658020, -133.3694153, 71.1658020, -204.5352173, 204.5352173
4: -123.0269012, 94.8894196, -123.0269012, 94.8894196, -217.9162903, 217.9162903
5: -109.9164963, 85.7271194, -109.9164963, 85.7271194, -195.6435394, 195.6435394
6: -105.6988068, 102.2890015, -105.6988068, 102.2890015, -207.9878082, 207.9878082
7: -114.6357803, 96.7082825, -114.6357803, 96.7082825, -211.3440399, 211.3440399
8: -139.8174286, 96.9648514, -139.8174286, 96.9648514, -236.7822723, 236.7822723
9: -104.8073578, 104.1193542, -104.8073578, 104.1193542, -208.9267120, 208.9267120

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5933138, upper bound: 202.5931069
time: 6.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5927550, upper bound: 202.5927093
time: 7.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -114.3123093, 91.4359589, -117.8387451, 94.2534485, -208.5657654, 209.2747040
1: -97.3022995, 81.0540771, -100.2893982, 83.5374603, -180.8397522, 181.3434753
2: -126.5931625, 82.2796860, -130.5082703, 84.7994843, -211.3926239, 212.7879486
3: -133.3694153, 71.1658020, -137.5389099, 73.3317947, -206.7012024, 208.7047119
4: -123.0269012, 94.8894196, -126.8203888, 97.7702942, -220.7971649, 221.7098083
5: -109.9164963, 85.7271194, -113.3322678, 88.3442764, -198.2607574, 199.0593262
6: -105.6988068, 102.2890015, -108.9602051, 105.4201050, -211.1189117, 211.2491760
7: -114.6357803, 96.7082825, -118.1710663, 99.6975708, -214.3333282, 214.8793488
8: -139.8174286, 96.9648514, -144.1304474, 99.8603821, -239.6778107, 241.0952911
9: -104.8073578, 104.1193542, -108.0024261, 107.2640152, -212.0713806, 212.1217804

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5933138, upper bound: 202.5931653
time: 6.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5927550, upper bound: 202.5927194
time: 7.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -117.8387451, 94.2534485, -118.3217239, 94.6082230, -212.4469604, 212.5751495
1: -100.2893982, 83.5374603, -100.6530533, 83.8545303, -184.1439209, 184.1904907
2: -130.5082703, 84.7994843, -131.0516357, 85.1679459, -215.6762085, 215.8510895
3: -137.5389099, 73.3317947, -138.1220856, 73.6486511, -211.1875458, 211.4538879
4: -126.8203888, 97.7702942, -127.3724747, 98.1757660, -224.9961548, 225.1427612
5: -113.3322678, 88.3442764, -113.8113022, 88.6931686, -202.0254211, 202.1555786
6: -108.9602051, 105.4201050, -109.3999405, 105.8280792, -214.7882843, 214.8200226
7: -118.1710663, 99.6975708, -118.7304840, 100.0825958, -218.2536621, 218.4280396
8: -144.1304474, 99.8603821, -144.6624298, 100.2747040, -244.4051514, 244.5228119
9: -108.0024261, 107.2640152, -108.4580841, 107.7458496, -215.7482758, 215.7221069

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5898125, upper bound: 202.5896635
time: 6.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5889328, upper bound: 202.5889663
time: 6.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -117.6941910, 94.1392746, -121.0371323, 96.7720108, -214.4662018, 215.1764069
1: -100.1677475, 83.4354553, -102.9319611, 85.7724915, -185.9402466, 186.3674164
2: -130.3497009, 84.6964111, -134.0450134, 87.0942993, -217.4440002, 218.7414093
3: -137.3681488, 73.2419052, -141.3194427, 75.3261566, -212.6943054, 214.5613403
4: -126.6662827, 97.6517029, -130.3013916, 100.4048157, -227.0710907, 227.9530945
5: -113.1922455, 88.2367706, -116.4254150, 90.7271118, -203.9193573, 204.6621857
6: -108.8267288, 105.2930603, -111.8944550, 108.2337799, -217.0604858, 217.1875153
7: -118.0269852, 99.5760956, -121.4392624, 102.3634796, -220.3904724, 221.0153351
8: -143.9565277, 99.7434921, -147.9403076, 102.5011215, -246.4576416, 247.6837769
9: -107.8715057, 107.1342926, -110.9288254, 110.1725922, -218.0440979, 218.0631104

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5900515, upper bound: 202.5898708
time: 7.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5891929, upper bound: 202.5891929
time: 7.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -114.3123093, 91.4359589, -116.9979630, 93.5887604, -207.9010468, 208.4339294
1: -97.3022995, 81.0540771, -99.5721359, 82.9556580, -180.2579651, 180.6262207
2: -126.5931625, 82.2796860, -129.5791321, 84.2278137, -210.8209686, 211.8588104
3: -133.3694153, 71.1658020, -136.5052948, 72.8001862, -206.1695862, 207.6710968
4: -123.0269012, 94.8894196, -125.9418793, 97.1169586, -220.1438599, 220.8312836
5: -109.9164963, 85.7271194, -112.5123062, 87.7526932, -197.6691437, 198.2393799
6: -105.6988068, 102.2890015, -108.1777878, 104.6804886, -210.3792877, 210.4667816
7: -114.6357803, 96.7082825, -117.3524323, 99.0018997, -213.6376495, 214.0606995
8: -139.8174286, 96.9648514, -143.1133118, 99.1928787, -239.0103149, 240.0781555
9: -104.8073578, 104.1193542, -107.2974091, 106.5389862, -211.3463440, 211.4167633

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5929852, upper bound: 202.5925857
time: 8.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5924586, upper bound: 202.5921900
time: 7.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -114.3123093, 91.4359589, -120.5904770, 96.4575043, -210.7697906, 212.0264282
1: -97.3022995, 81.0540771, -102.6173248, 85.4852600, -182.7875519, 183.6714020
2: -126.5931625, 82.2796860, -133.5718536, 86.7935028, -213.3866577, 215.8515320
3: -133.3694153, 71.1658020, -140.7533112, 75.0046387, -208.3740540, 211.9191132
4: -123.0269012, 94.8894196, -129.8094025, 100.0519791, -223.0788727, 224.6988068
5: -109.9164963, 85.7271194, -115.9898376, 90.4180756, -200.3345490, 201.7169189
6: -105.6988068, 102.2890015, -111.4987869, 107.8731232, -213.5719299, 213.7877808
7: -114.6357803, 96.7082825, -120.9549789, 102.0482559, -216.6840363, 217.6632385
8: -139.8174286, 96.9648514, -147.5045624, 102.1433716, -241.9607849, 244.4694061
9: -104.8073578, 104.1193542, -110.5523911, 109.7455215, -214.5528870, 214.6717377

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5929852, upper bound: 202.5926208
time: 7.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5924586, upper bound: 202.5922039
time: 8.18 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -117.8387451, 94.2534485, -120.9748306, 96.7351913, -214.5739288, 215.2282715
1: -100.2893982, 83.5374603, -102.8970871, 85.7334213, -186.0228271, 186.4345398
2: -130.5082703, 84.7994843, -134.0041809, 87.0956650, -217.6039276, 218.8036346
3: -137.5389099, 73.3317947, -141.2195129, 75.2622375, -212.8011475, 214.5513000
4: -126.8203888, 97.7702942, -130.2546387, 100.3762741, -227.1966553, 228.0249023
5: -113.3322678, 88.3442764, -116.3755035, 90.6940536, -204.0263214, 204.7197876
6: -108.9602051, 105.4201050, -111.8494797, 108.1939087, -217.1540833, 217.2695770
7: -118.1710663, 99.6975708, -121.4175339, 102.3515396, -220.5226135, 221.1151123
8: -144.1304474, 99.8603821, -147.9215240, 102.4754486, -246.6058960, 247.7819061
9: -108.0024261, 107.2640152, -110.9185181, 110.1370468, -218.1394653, 218.1825256

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5894180, upper bound: 202.5890547
time: 7.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5885889, upper bound: 202.5883673
time: 7.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -117.6941910, 94.1392746, -124.3038483, 99.3881836, -217.0823517, 218.4431152
1: -100.1677475, 83.4354553, -105.6941833, 88.0833206, -188.2510681, 189.1296387
2: -130.3497009, 84.6964111, -137.6775970, 89.4531097, -219.8028107, 222.3740082
3: -137.3681488, 73.2419052, -145.1398926, 77.3160324, -214.6841736, 218.3817902
4: -126.6662827, 97.6517029, -133.8400116, 103.1091003, -229.7753754, 231.4917145
5: -113.1922455, 88.2367706, -119.5862579, 93.1845169, -206.3767548, 207.8230133
6: -108.8267288, 105.2930603, -114.9049835, 111.1417542, -219.9684753, 220.1980438
7: -118.0269852, 99.5760956, -124.7388077, 105.1457825, -223.1727600, 224.3148956
8: -143.9565277, 99.7434921, -151.9369202, 105.2023010, -249.1588287, 251.6803894
9: -107.8715057, 107.1342926, -113.9486313, 113.1136398, -220.9851227, 221.0829010

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5898709, upper bound: 202.5895246
time: 7.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5890152, upper bound: 202.5888404
time: 8.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -116.9979630, 93.5887604, -114.3123093, 91.4359589, -208.4339294, 207.9010468
1: -99.5721359, 82.9556580, -97.3022995, 81.0540771, -180.6262207, 180.2579651
2: -129.5791321, 84.2278137, -126.5931625, 82.2796860, -211.8588104, 210.8209686
3: -136.5052948, 72.8001862, -133.3694153, 71.1658020, -207.6710968, 206.1695862
4: -125.9418793, 97.1169586, -123.0269012, 94.8894196, -220.8312836, 220.1438599
5: -112.5123062, 87.7526932, -109.9164963, 85.7271194, -198.2393799, 197.6691437
6: -108.1777878, 104.6804886, -105.6988068, 102.2890015, -210.4667816, 210.3792877
7: -117.3524323, 99.0018997, -114.6357803, 96.7082825, -214.0606995, 213.6376495
8: -143.1133118, 99.1928787, -139.8174286, 96.9648514, -240.0781555, 239.0103149
9: -107.2974091, 106.5389862, -104.8073578, 104.1193542, -211.4167633, 211.3463440

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5928580, upper bound: 202.5928630
time: 8.00 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5921851, upper bound: 202.5923614
time: 7.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -116.9979630, 93.5887604, -117.8387451, 94.2534485, -211.2514038, 211.4274750
1: -99.5721359, 82.9556580, -100.2893982, 83.5374603, -183.1095734, 183.2450562
2: -129.5791321, 84.2278137, -130.5082703, 84.7994843, -214.3785858, 214.7360687
3: -136.5052948, 72.8001862, -137.5389099, 73.3317947, -209.8370972, 210.3390656
4: -125.9418793, 97.1169586, -126.8203888, 97.7702942, -223.7121582, 223.9373474
5: -112.5123062, 87.7526932, -113.3322678, 88.3442764, -200.8565826, 201.0849304
6: -108.1777878, 104.6804886, -108.9602051, 105.4201050, -213.5979004, 213.6406708
7: -117.3524323, 99.0018997, -118.1710663, 99.6975708, -217.0500031, 217.1729431
8: -143.1133118, 99.1928787, -144.1304474, 99.8603821, -242.9736938, 243.3233337
9: -107.2974091, 106.5389862, -108.0024261, 107.2640152, -214.5614319, 214.5414124

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5928580, upper bound: 202.5929228
time: 7.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5921851, upper bound: 202.5923837
time: 7.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -120.5904770, 96.4575043, -118.3217239, 94.6082230, -215.1987000, 214.7791748
1: -102.6173248, 85.4852600, -100.6530533, 83.8545303, -186.4718628, 186.1382904
2: -133.5718536, 86.7935028, -131.0516357, 85.1679459, -218.7398071, 217.8451385
3: -140.7533112, 75.0046387, -138.1220856, 73.6486511, -214.4019623, 213.1267242
4: -129.8094025, 100.0519791, -127.3724747, 98.1757660, -227.9851685, 227.4244537
5: -115.9898376, 90.4180756, -113.8113022, 88.6931686, -204.6830139, 204.2293701
6: -111.4987869, 107.8731232, -109.3999405, 105.8280792, -217.3268738, 217.2730713
7: -120.9549789, 102.0482559, -118.7304840, 100.0825958, -221.0375519, 220.7787476
8: -147.5045624, 102.1433716, -144.6624298, 100.2747040, -247.7792664, 246.8057861
9: -110.5523911, 109.7455215, -108.4580841, 107.7458496, -218.2982330, 218.2035980

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5896188, upper bound: 202.5895671
time: 7.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5885389, upper bound: 202.5887573
time: 7.96 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -120.4447556, 96.3422394, -121.0371323, 96.7720108, -217.2167664, 217.3793640
1: -102.4946289, 85.3824387, -102.9319611, 85.7724915, -188.2671204, 188.3143921
2: -133.4118042, 86.6894608, -134.0450134, 87.0942993, -220.5061035, 220.7344666
3: -140.5811462, 74.9141235, -141.3194427, 75.3261566, -215.9073029, 216.2335663
4: -129.6539764, 99.9323730, -130.3013916, 100.4048157, -230.0587921, 230.2337646
5: -115.8487244, 90.3097076, -116.4254150, 90.7271118, -206.5758209, 206.7351227
6: -111.3641510, 107.7449722, -111.8944550, 108.2337799, -219.5979156, 219.6394348
7: -120.8096924, 101.9258423, -121.4392624, 102.3634796, -223.1731720, 223.3650513
8: -147.3289032, 102.0250854, -147.9403076, 102.5011215, -249.8300171, 249.9653931
9: -110.4204865, 109.6147003, -110.9288254, 110.1725922, -220.5930786, 220.5435181

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5899215, upper bound: 202.5898129
time: 7.27 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5888404, upper bound: 202.5890152
time: 7.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -116.9979630, 93.5887604, -116.9979630, 93.5887604, -210.5867157, 210.5867157
1: -99.5721359, 82.9556580, -99.5721359, 82.9556580, -182.5278015, 182.5278015
2: -129.5791321, 84.2278137, -129.5791321, 84.2278137, -213.8069305, 213.8069305
3: -136.5052948, 72.8001862, -136.5052948, 72.8001862, -209.3054810, 209.3054810
4: -125.9418793, 97.1169586, -125.9418793, 97.1169586, -223.0588379, 223.0588379
5: -112.5123062, 87.7526932, -112.5123062, 87.7526932, -200.2649841, 200.2649841
6: -108.1777878, 104.6804886, -108.1777878, 104.6804886, -212.8582764, 212.8582764
7: -117.3524323, 99.0018997, -117.3524323, 99.0018997, -216.3543091, 216.3543091
8: -143.1133118, 99.1928787, -143.1133118, 99.1928787, -242.3061829, 242.3061829
9: -107.2974091, 106.5389862, -107.2974091, 106.5389862, -213.8363953, 213.8363953

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5927271, upper bound: 202.5925160
time: 7.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5921199, upper bound: 202.5921016
time: 7.17 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -116.9979630, 93.5887604, -120.5904770, 96.4575043, -213.4554596, 214.1792297
1: -99.5721359, 82.9556580, -102.6173248, 85.4852600, -185.0573730, 185.5729675
2: -129.5791321, 84.2278137, -133.5718536, 86.7935028, -216.3726349, 217.7996521
3: -136.5052948, 72.8001862, -140.7533112, 75.0046387, -211.5099335, 213.5534821
4: -125.9418793, 97.1169586, -129.8094025, 100.0519791, -225.9938660, 226.9263611
5: -112.5123062, 87.7526932, -115.9898376, 90.4180756, -202.9303894, 203.7425232
6: -108.1777878, 104.6804886, -111.4987869, 107.8731232, -216.0509033, 216.1792603
7: -117.3524323, 99.0018997, -120.9549789, 102.0482559, -219.4006958, 219.9568481
8: -143.1133118, 99.1928787, -147.5045624, 102.1433716, -245.2566833, 246.6974487
9: -107.2974091, 106.5389862, -110.5523911, 109.7455215, -217.0429382, 217.0913544

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5927271, upper bound: 202.5925659
time: 7.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5921199, upper bound: 202.5921216
time: 7.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -120.5904770, 96.4575043, -120.9748306, 96.7351913, -217.3256683, 217.4322968
1: -102.6173248, 85.4852600, -102.8970871, 85.7334213, -188.3507385, 188.3823395
2: -133.5718536, 86.7935028, -134.0041809, 87.0956650, -220.6675110, 220.7976837
3: -140.7533112, 75.0046387, -141.2195129, 75.2622375, -216.0155487, 216.2241516
4: -129.8094025, 100.0519791, -130.2546387, 100.3762741, -230.1856689, 230.3066101
5: -115.9898376, 90.4180756, -116.3755035, 90.6940536, -206.6838989, 206.7935791
6: -111.4987869, 107.8731232, -111.8494797, 108.1939087, -219.6926880, 219.7225952
7: -120.9549789, 102.0482559, -121.4175339, 102.3515396, -223.3065186, 223.4657898
8: -147.5045624, 102.1433716, -147.9215240, 102.4754486, -249.9800110, 250.0648956
9: -110.5523911, 109.7455215, -110.9185181, 110.1370468, -220.6894226, 220.6640320

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5894045, upper bound: 202.5891404
time: 7.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5883680, upper bound: 202.5883181
time: 7.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -120.4447556, 96.3422394, -124.3038483, 99.3881836, -219.8329468, 220.6460876
1: -102.4946289, 85.3824387, -105.6941833, 88.0833206, -190.5779266, 191.0765991
2: -133.4118042, 86.6894608, -137.6775970, 89.4531097, -222.8649139, 224.3670654
3: -140.5811462, 74.9141235, -145.1398926, 77.3160324, -217.8971863, 220.0540161
4: -129.6539764, 99.9323730, -133.8400116, 103.1091003, -232.7630768, 233.7723846
5: -115.8487244, 90.3097076, -119.5862579, 93.1845169, -209.0332336, 209.8959351
6: -111.3641510, 107.7449722, -114.9049835, 111.1417542, -222.5059052, 222.6499634
7: -120.8096924, 101.9258423, -124.7388077, 105.1457825, -225.9554749, 226.6646271
8: -147.3289032, 102.0250854, -151.9369202, 105.2023010, -252.5311890, 253.9620056
9: -110.4204865, 109.6147003, -113.9486313, 113.1136398, -223.5341034, 223.5633087

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5898620, upper bound: 202.5896007
time: 6.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5887817, upper bound: 202.5887907
time: 6.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.30 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5933138, upper bound: 202.5931069
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5927550, upper bound: 202.5927093
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5933138, upper bound: 202.5931653
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5927550, upper bound: 202.5927194
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5898125, upper bound: 202.5896635
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5889328, upper bound: 202.5889663
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5900515, upper bound: 202.5898708
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5891929, upper bound: 202.5891929
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5929852, upper bound: 202.5925857
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5924586, upper bound: 202.5921900
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5929852, upper bound: 202.5926208
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5924586, upper bound: 202.5922039
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5894180, upper bound: 202.5890547
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5885889, upper bound: 202.5883673
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5898709, upper bound: 202.5895246
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5890152, upper bound: 202.5888404
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5928580, upper bound: 202.5928630
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5921851, upper bound: 202.5923614
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5928580, upper bound: 202.5929228
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5921851, upper bound: 202.5923837
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5896188, upper bound: 202.5895671
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5885389, upper bound: 202.5887573
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5899215, upper bound: 202.5898129
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5888404, upper bound: 202.5890152
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5927271, upper bound: 202.5925160
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5921199, upper bound: 202.5921016
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5927271, upper bound: 202.5925659
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5921199, upper bound: 202.5921216
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5894045, upper bound: 202.5891404
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5883680, upper bound: 202.5883181
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5898620, upper bound: 202.5896007
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.30
Output dim: 1, lower bound: -202.5887817, upper bound: 202.5887907

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -107.6699219, 86.1681442, -114.3123093, 91.4359589, -199.1058807, 200.4804535
1: -91.7614746, 76.3675842, -97.3022995, 81.0540771, -172.8155518, 173.6698914
2: -119.2643814, 77.5086975, -126.5931625, 82.2796860, -201.5440674, 204.1018677
3: -125.4551544, 67.0431824, -133.3694153, 71.1658020, -196.6209564, 200.4125824
4: -115.8620224, 89.4609451, -123.0269012, 94.8894196, -210.7514038, 212.4878387
5: -103.4877777, 80.7284012, -109.9164963, 85.7271194, -189.2148438, 190.6448669
6: -99.5756836, 96.4111404, -105.6988068, 102.2890015, -201.8646851, 202.1099548
7: -107.9452667, 91.1029816, -114.6357803, 96.7082825, -204.6535492, 205.7387543
8: -131.8758240, 91.5710983, -139.8174286, 96.9648514, -228.8406677, 231.3885193
9: -98.7694931, 98.1429367, -104.8073578, 104.1193542, -202.8888550, 202.9502869

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5931537, upper bound: 202.5931537
time: 6.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5931537, upper bound: 202.5931537
time: 7.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -110.5131607, 88.4566498, -113.4144669, 90.7257767, -201.2389069, 201.8710938
1: -94.1543427, 78.3518219, -96.5523911, 80.4206696, -174.5750122, 174.9042053
2: -122.3831635, 79.4913635, -125.6021042, 81.6331253, -204.0162964, 205.0934601
3: -128.7768555, 68.7723160, -132.3002625, 70.6103592, -199.3872070, 201.0725708
4: -118.8952713, 91.7470016, -122.0566483, 94.1528015, -213.0480652, 213.8036346
5: -106.2246552, 82.8105850, -109.0494843, 85.0508118, -191.2754669, 191.8600464
6: -102.1944580, 98.9261551, -104.8709564, 101.4932861, -203.6877289, 203.7971191
7: -110.7637329, 93.4904938, -113.7295151, 95.9502029, -206.7139282, 207.2200012
8: -135.3586273, 93.9141006, -138.7429199, 96.2363815, -231.5950012, 232.6570129
9: -101.3353653, 100.6921616, -103.9879150, 103.3100510, -204.6454163, 204.6800842

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5893455, upper bound: 202.5893733
time: 7.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5894919, upper bound: 202.5894919
time: 6.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -107.6699219, 86.1681442, -117.8387451, 94.2534485, -201.9233704, 204.0068817
1: -91.7614746, 76.3675842, -100.2893982, 83.5374603, -175.2989349, 176.6569824
2: -119.2643814, 77.5086975, -130.5082703, 84.7994843, -204.0638428, 208.0169678
3: -125.4551544, 67.0431824, -137.5389099, 73.3317947, -198.7869568, 204.5820618
4: -115.8620224, 89.4609451, -126.8203888, 97.7702942, -213.6322632, 216.2813416
5: -103.4877777, 80.7284012, -113.3322678, 88.3442764, -191.8320618, 194.0606689
6: -99.5756836, 96.4111404, -108.9602051, 105.4201050, -204.9957886, 205.3713379
7: -107.9452667, 91.1029816, -118.1710663, 99.6975708, -207.6428223, 209.2740479
8: -131.8758240, 91.5710983, -144.1304474, 99.8603821, -231.7362061, 235.7015381
9: -98.7694931, 98.1429367, -108.0024261, 107.2640152, -206.0335083, 206.1453552

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5927557, upper bound: 202.5926829
time: 6.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5927557, upper bound: 202.5926829
time: 6.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -110.5131607, 88.4566498, -116.9364700, 93.5396042, -204.0527496, 205.3930969
1: -94.1543427, 78.3518219, -99.5355301, 82.9008789, -177.0552063, 177.8873596
2: -122.3831635, 79.4913635, -129.5123291, 84.1497955, -206.5329590, 209.0036774
3: -128.7768555, 68.7723160, -136.4643250, 72.7731476, -201.5500031, 205.2366333
4: -118.8952713, 91.7470016, -125.8456497, 97.0301743, -215.9253998, 217.5926514
5: -106.2246552, 82.8105850, -112.4607086, 87.6645966, -193.8892517, 195.2712402
6: -102.1944580, 98.9261551, -108.1281281, 104.6203461, -206.8148041, 207.0542908
7: -110.7637329, 93.4904938, -117.2605515, 98.9356003, -209.6993408, 210.7510376
8: -135.3586273, 93.9141006, -143.0505676, 99.1285400, -234.4871674, 236.9646606
9: -101.3353653, 100.6921616, -107.1789856, 106.4502716, -207.7856293, 207.8711395

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5888809, upper bound: 202.5888859
time: 7.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5890987, upper bound: 202.5890679
time: 7.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -111.2619553, 89.0385590, -118.3217239, 94.6082230, -205.8701782, 207.3602600
1: -94.8042603, 78.8972397, -100.6530533, 83.8545303, -178.6587830, 179.5502777
2: -123.2526474, 80.0755234, -131.0516357, 85.1679459, -208.4205933, 211.1271362
3: -129.7076874, 69.2502594, -138.1220856, 73.6486511, -203.3563385, 207.3723145
4: -119.7265625, 92.3954773, -127.3724747, 98.1757660, -217.9023285, 219.7679443
5: -106.9698486, 83.3942947, -113.8113022, 88.6931686, -195.6630249, 197.2055817
6: -102.8991776, 99.6005096, -109.3999405, 105.8280792, -208.7272644, 209.0004425
7: -111.5461349, 94.1484222, -118.7304840, 100.0825958, -211.6287231, 212.8789062
8: -136.2673187, 94.5180969, -144.6624298, 100.2747040, -236.5420227, 239.1805115
9: -102.0229340, 101.3476639, -108.4580841, 107.7458496, -209.7687836, 209.8057556

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5889314, upper bound: 202.5889648
time: 7.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5889314, upper bound: 202.5889648
time: 7.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -113.6908340, 90.9972305, -117.4030151, 93.8810425, -207.5718384, 208.4002380
1: -96.8477783, 80.5922089, -99.8852158, 83.2063522, -180.0541077, 180.4774170
2: -125.9146729, 81.7593765, -130.0370789, 84.5062866, -210.4209442, 211.7964478
3: -132.5363770, 70.7241211, -137.0285492, 73.0801620, -205.6165466, 207.7526703
4: -122.3153152, 94.3435440, -126.3801422, 97.4216690, -219.7369843, 220.7236633
5: -109.3021088, 85.1687546, -112.9238968, 88.0012283, -197.3033447, 198.0926514
6: -105.1361084, 101.7490692, -108.5532379, 105.0134430, -210.1495361, 210.3022614
7: -113.9491119, 96.1838684, -117.8028946, 99.3066330, -213.2557373, 213.9867554
8: -139.2495270, 96.5252686, -143.5626221, 99.5284195, -238.7779388, 240.0878906
9: -104.2135086, 103.5220490, -107.6195374, 106.9172974, -211.1307983, 211.1415710

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5887914, upper bound: 202.5887231
time: 7.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5887914, upper bound: 202.5889663
time: 7.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -111.1167679, 88.9238968, -121.0371323, 96.7720108, -207.8887787, 209.9610291
1: -94.6821136, 78.7947998, -102.9319611, 85.7724915, -180.4546051, 181.7267609
2: -123.0933762, 79.9720230, -134.0450134, 87.0942993, -210.1876831, 214.0170288
3: -129.5362701, 69.1599655, -141.3194427, 75.3261566, -204.8624268, 210.4794006
4: -119.5717926, 92.2763748, -130.3013916, 100.4048157, -219.9766083, 222.5777588
5: -106.8292007, 83.2863083, -116.4254150, 90.7271118, -197.5563049, 199.7117310
6: -102.7651062, 99.4729462, -111.8944550, 108.2337799, -210.9988708, 211.3673706
7: -111.4014816, 94.0264206, -121.4392624, 102.3634796, -213.7649536, 215.4656677
8: -136.0926208, 94.4006882, -147.9403076, 102.5011215, -238.5937500, 242.3410034
9: -101.8914719, 101.2173691, -110.9288254, 110.1725922, -212.0640564, 212.1461945

Time for backsubstitution: 1.07 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=203.5233612060547
rel_dist={1: [-202.60902678108835, 202.60902678108835]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6048844, upper bound: 202.6047430
time: 8.84 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6047343, upper bound: 202.6047343
time: 8.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.53 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.53
Output dim: 1, lower bound: -202.6048844, upper bound: 202.6047430
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.53
Output dim: 1, lower bound: -202.6047343, upper bound: 202.6047343

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -123.0395737, 98.3348694, -128.6298370, 102.7676315, -225.8071594, 226.9646912
1: -104.6226501, 87.1827240, -109.3534470, 91.1268539, -195.7495117, 196.5361633
2: -136.2332764, 88.5248413, -142.4177551, 92.5298004, -228.7630615, 230.9425964
3: -143.6921082, 76.5960464, -150.2513580, 80.0430908, -223.7351990, 226.8473969
4: -132.4142151, 102.0421982, -138.4431610, 106.6591339, -239.0733337, 240.4853516
5: -118.3712463, 92.2120056, -123.7401047, 96.3995056, -214.7707062, 215.9521179
6: -113.7487183, 109.9841461, -118.8830261, 114.9493103, -228.6980286, 228.8671722
7: -123.4394455, 104.0390549, -129.0890656, 108.7745895, -232.2140350, 233.1281128
8: -150.3356781, 104.1154709, -157.0911407, 108.7183380, -259.0539856, 261.2066040
9: -112.7398911, 111.9838409, -117.8671494, 117.0114365, -229.7513275, 229.8509827

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5987736, upper bound: 202.5986473
time: 8.19 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5992647, upper bound: 202.5991821
time: 8.00 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.7521286, 100.5080948, -128.8923035, 102.9804993, -228.7326355, 229.4003906
1: -106.9180679, 89.1028976, -109.5783920, 91.3141479, -198.2321930, 198.6812592
2: -139.2514191, 90.4931870, -142.7142944, 92.7231369, -231.9745178, 233.2074890
3: -146.8589630, 78.2464371, -150.5551910, 80.2021561, -227.0610809, 228.8016357
4: -135.3592224, 104.2918320, -138.7328339, 106.8809891, -242.2401733, 243.0246582
5: -120.9924850, 94.2561188, -123.9917221, 96.6015244, -217.5940094, 218.2478333
6: -116.2524948, 112.4022141, -119.1269379, 115.1870499, -231.4395294, 231.5291443
7: -126.1867523, 106.3574982, -129.3626404, 109.0055847, -235.1923218, 235.7201385
8: -153.6653137, 106.3638687, -157.4187622, 108.9366226, -262.6018677, 263.7826233
9: -115.2547913, 114.4284286, -118.1207199, 117.2546082, -232.5093994, 232.5491486

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5987137, upper bound: 202.5986682
time: 9.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5992299, upper bound: 202.5992299
time: 8.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.47 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.47
Output dim: 1, lower bound: -202.5987736, upper bound: 202.5986473
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.47
Output dim: 1, lower bound: -202.5992647, upper bound: 202.5991821
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.47
Output dim: 1, lower bound: -202.5987137, upper bound: 202.5986682
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.47
Output dim: 1, lower bound: -202.5992299, upper bound: 202.5992299

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -120.1306763, 96.0358505, -119.9192657, 95.8822403, -216.0129089, 215.9550934
1: -102.1820450, 85.1399155, -102.0445099, 85.0098953, -187.1918945, 187.1844177
2: -133.0195312, 86.4435577, -132.7945099, 86.2969360, -219.3164673, 219.2380676
3: -140.2511597, 74.7852402, -139.9483032, 74.6231308, -214.8742828, 214.7335205
4: -129.2850647, 99.6582260, -129.0730743, 99.5191803, -228.8042450, 228.7312927
5: -115.5536194, 90.0511627, -115.3015213, 89.9286575, -205.4822693, 205.3526917
6: -111.0661697, 107.4183884, -110.8485489, 107.2667618, -218.3329163, 218.2669373
7: -120.5044327, 101.5958939, -120.3012543, 101.4572983, -221.9617310, 221.8971558
8: -146.8298645, 101.7317352, -146.5921326, 101.5814667, -248.4113312, 248.3238678
9: -110.0955582, 109.3627396, -109.9487381, 109.1612778, -219.2568207, 219.3114777

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5950443, upper bound: 202.5948874
time: 8.21 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5953846, upper bound: 202.5952521
time: 8.35 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -119.6241226, 95.6389771, -123.4435120, 98.6979294, -218.3220367, 219.0824890
1: -101.7582855, 84.7831421, -105.0313644, 87.4918671, -189.2501526, 189.8145142
2: -132.4667206, 86.0864792, -136.7104034, 88.8146133, -221.2813263, 222.7968750
3: -139.6604309, 74.4752121, -144.1151428, 76.7858429, -216.4462738, 218.5903473
4: -128.7326050, 99.2358246, -132.8652649, 102.3986893, -231.1312714, 232.1010895
5: -115.0706711, 89.6726685, -118.7138596, 92.5426788, -207.6133118, 208.3865356
6: -110.6008148, 106.9700851, -114.1075363, 110.3967514, -220.9975433, 221.0775757
7: -119.9927368, 101.1779480, -123.8332901, 104.4449005, -224.4376373, 225.0112305
8: -146.2220459, 101.3148499, -150.9014587, 104.4769897, -250.6990051, 252.2162781
9: -109.6257477, 108.8939438, -113.1427002, 112.3045349, -221.9302826, 222.0366516

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5954612, upper bound: 202.5953572
time: 8.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5958612, upper bound: 202.5957919
time: 8.00 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -122.8257751, 98.1954041, -120.1590652, 96.0775757, -218.9033508, 218.3544312
1: -104.4614944, 87.0478897, -102.2497559, 85.1810455, -189.6425323, 189.2976074
2: -136.0170746, 88.3990784, -133.0653992, 86.4737091, -222.4907837, 221.4644775
3: -143.3976440, 76.4252014, -140.2262573, 74.7683258, -218.1659698, 216.6514282
4: -132.2107544, 101.8938751, -129.3384247, 99.7228317, -231.9335785, 231.2322845
5: -118.1577606, 92.0834274, -115.5319061, 90.1138611, -208.2716217, 207.6153259
6: -113.5534439, 109.8196869, -111.0720291, 107.4835663, -221.0370026, 220.8917236
7: -123.2328262, 103.8988266, -120.5507584, 101.6686554, -224.9014893, 224.4495850
8: -150.1379242, 103.9667740, -146.8919220, 101.7819366, -251.9198608, 250.8587036
9: -112.5945129, 111.7915039, -110.1820374, 109.3837967, -221.9783020, 221.9735413

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5919332, upper bound: 202.5918071
time: 8.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5913944, upper bound: 202.5914029
time: 8.91 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -122.3221741, 97.7997284, -123.7158661, 98.9183807, -221.2405243, 221.5155945
1: -104.0411224, 86.6926346, -105.2652206, 87.6857910, -191.7269135, 191.9578552
2: -135.4689026, 88.0438080, -137.0184326, 89.0143433, -224.4832458, 225.0622406
3: -142.8090820, 76.1158371, -144.4313965, 76.9506302, -219.7597046, 220.5472107
4: -131.6614380, 101.4732132, -133.1665802, 102.6288376, -234.2902679, 234.6398010
5: -117.6764297, 91.7072296, -118.9754562, 92.7524109, -210.4288177, 210.6826782
6: -113.0903854, 109.3745117, -114.3608932, 110.6437225, -223.7340546, 223.7353821
7: -122.7247849, 103.4844818, -124.1165009, 104.6846237, -227.4094086, 227.6009827
8: -149.5328979, 103.5513687, -151.2402802, 104.7039032, -254.2368011, 254.7916565
9: -112.1263657, 111.3259506, -113.4052582, 112.5574493, -224.6837616, 224.7312012

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5921060, upper bound: 202.5919899
time: 10.41 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5915195, upper bound: 202.5915195
time: 8.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.72 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.72
Output dim: 1, lower bound: -202.5950443, upper bound: 202.5948874
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.72
Output dim: 1, lower bound: -202.5953846, upper bound: 202.5952521
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.72
Output dim: 1, lower bound: -202.5954612, upper bound: 202.5953572
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.72
Output dim: 1, lower bound: -202.5958612, upper bound: 202.5957919
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.72
Output dim: 1, lower bound: -202.5919332, upper bound: 202.5918071
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.72
Output dim: 1, lower bound: -202.5913944, upper bound: 202.5914029
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.72
Output dim: 1, lower bound: -202.5921060, upper bound: 202.5919899
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.72
Output dim: 1, lower bound: -202.5915195, upper bound: 202.5915195

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -116.0604248, 92.8197327, -118.5205841, 94.7764435, -210.8368683, 211.3403015
1: -98.7558594, 82.2672882, -100.8676529, 84.0233154, -182.7791443, 183.1349335
2: -128.5520630, 83.5479507, -131.2588348, 85.3020935, -213.8541565, 214.8067932
3: -135.4460907, 72.2430878, -138.2969971, 73.7509232, -209.1969910, 210.5400848
4: -124.9413300, 96.3229828, -127.5805664, 98.3726730, -223.3139954, 223.9035492
5: -111.6186676, 87.0133743, -113.9491043, 88.8848572, -200.5035248, 200.9624786
6: -107.3137817, 103.8352203, -109.5591507, 106.0360794, -213.3498535, 213.3943329
7: -116.4493256, 98.1809845, -118.9076920, 100.2836227, -216.7329407, 217.0886688
8: -141.9355774, 98.4229660, -144.9096680, 100.4454575, -242.3810425, 243.3325958
9: -106.4047318, 105.7097931, -108.6808014, 107.9055634, -214.3102875, 214.3905945

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5880708, upper bound: 202.5880256
time: 9.49 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5877189, upper bound: 202.5875967
time: 8.54 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -118.7915497, 94.9964218, -118.5663223, 94.8126144, -213.6041565, 213.5627441
1: -101.0479584, 84.1960220, -100.9054565, 84.0550766, -185.1030121, 185.1014557
2: -131.5623322, 85.4852219, -131.3087769, 85.3315430, -216.8938751, 216.7939911
3: -138.6616669, 73.9296646, -138.3499603, 73.7820206, -212.4436951, 212.2796173
4: -127.8868332, 98.5646515, -127.6297684, 98.4088058, -226.2956390, 226.1943970
5: -114.2478256, 89.0589828, -113.9911728, 88.9221268, -203.1699524, 203.0501404
6: -109.8222809, 106.2543182, -109.5985641, 106.0768738, -215.8991547, 215.8528748
7: -119.1735916, 100.4749069, -118.9517441, 100.3207474, -219.4943390, 219.4266510
8: -145.2320709, 100.6629333, -144.9619598, 100.4832993, -245.7153625, 245.6248932
9: -108.8896484, 108.1501541, -108.7236404, 107.9459686, -216.8355865, 216.8737946

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5883775, upper bound: 202.5883351
time: 8.97 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5880191, upper bound: 202.5879419
time: 8.04 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -115.5462723, 92.4176865, -122.0300827, 97.5808487, -213.1271057, 214.4477692
1: -98.3259201, 81.9047012, -103.8418579, 86.4949265, -184.8208466, 185.7465515
2: -127.9906235, 83.1856308, -135.1582794, 87.8092422, -215.7998505, 218.3439026
3: -134.8462372, 71.9272537, -142.4468079, 75.9042435, -210.7504578, 214.3740540
4: -124.3808975, 95.8944626, -131.3569794, 101.2399597, -225.6208496, 227.2514191
5: -111.1287155, 86.6294479, -117.3472366, 91.4878082, -202.6165161, 203.9766846
6: -106.8415833, 103.3798981, -112.8045502, 109.1528244, -215.9943695, 216.1844482
7: -115.9298172, 97.7568512, -122.4248886, 103.2591705, -219.1889801, 220.1817322
8: -141.3194580, 98.0010605, -149.2011414, 103.3282700, -244.6477356, 247.2021942
9: -105.9277878, 105.2342529, -111.8618469, 111.0356979, -216.9634705, 217.0960846

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5884304, upper bound: 202.5884187
time: 7.81 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5879129, upper bound: 202.5877539
time: 8.31 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -118.2611160, 94.5804291, -122.0897751, 97.6270981, -215.8881836, 216.6701813
1: -100.6043320, 83.8225327, -103.8914642, 86.5366974, -187.1409607, 187.7139893
2: -130.9836121, 85.1117706, -135.2233887, 87.8485107, -218.8321228, 220.3351440
3: -138.0424042, 73.6048965, -142.5157471, 75.9450302, -213.9874268, 216.1206360
4: -127.3091278, 98.1229935, -131.4209595, 101.2872772, -228.5964050, 229.5439453
5: -113.7420578, 88.6627502, -117.4028168, 91.5357285, -205.2777863, 206.0655670
6: -109.3358917, 105.7848892, -112.8568726, 109.2058640, -218.5417480, 218.6417389
7: -118.6380081, 100.0366516, -122.4829254, 103.3075256, -221.9454956, 222.5195770
8: -144.5967407, 100.2278214, -149.2695618, 103.3782730, -247.9750061, 249.4973755
9: -108.3978119, 107.6604919, -111.9167938, 111.0888214, -219.4865875, 219.5772858

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5887699, upper bound: 202.5888090
time: 8.48 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5882699, upper bound: 202.5881462
time: 8.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -116.1269608, 92.8805084, -118.2917938, 94.5955582, -210.7225189, 211.1723022
1: -98.8737335, 82.3219681, -100.6912842, 83.8636246, -182.7373657, 183.0132141
2: -128.6261749, 83.5881500, -131.0046082, 85.1324844, -213.7586670, 214.5927582
3: -135.4197693, 72.2690964, -138.0014496, 73.6095886, -209.0293427, 210.2705383
4: -124.9887161, 96.4153671, -127.3249893, 98.1959381, -223.1846619, 223.7403564
5: -111.6729279, 87.0424805, -113.7242813, 88.7084656, -200.3813934, 200.7667542
6: -107.3824997, 103.8915634, -109.3519669, 105.8304825, -213.2129822, 213.2434998
7: -116.4867249, 98.2468414, -118.6697083, 100.0929794, -216.5796967, 216.9165497
8: -142.1268616, 98.5245819, -144.6587982, 100.2646179, -242.3914490, 243.1833801
9: -106.5061188, 105.7631836, -108.4850311, 107.7029419, -214.2090454, 214.2481995

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5902593, upper bound: 202.5901214
time: 8.73 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5902526, upper bound: 202.5901034
time: 10.33 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -118.9339523, 95.1458359, -116.6753769, 93.3202820, -212.2542267, 211.8211823
1: -101.2338791, 84.2820511, -99.3373413, 82.7230453, -183.9568939, 183.6193848
2: -131.7104340, 85.5416718, -129.2194214, 83.9644165, -215.6748352, 214.7610779
3: -138.7006378, 73.9794769, -136.0790405, 72.6125717, -211.3132019, 210.0585175
4: -127.9787216, 98.6738892, -125.5738602, 96.8633652, -224.8420868, 224.2477417
5: -114.3762131, 89.0911636, -112.1675644, 87.4881668, -201.8643799, 201.2587280
6: -109.9639816, 106.3762360, -107.8606186, 104.3947144, -214.3587036, 214.2368469
7: -119.2632828, 100.5977936, -117.0329285, 98.7264938, -217.9897308, 217.6307220
8: -145.5663910, 100.8397903, -142.7213287, 98.9516373, -244.5180359, 243.5610962
9: -109.0305176, 108.2798920, -107.0016403, 106.2419281, -215.2724457, 215.2815247

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5897519, upper bound: 202.5896567
time: 8.57 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5897466, upper bound: 202.5896518
time: 9.14 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -115.6567078, 92.5127106, -121.8685074, 97.4526367, -213.1093445, 214.3811951
1: -98.4816437, 81.9907684, -103.7236404, 86.3825455, -184.8641968, 185.7144165
2: -128.1154327, 83.2573318, -134.9795532, 87.6880188, -215.8034515, 218.2368622
3: -134.8731842, 71.9812469, -142.2315826, 75.8045731, -210.6777344, 214.2128296
4: -124.4757309, 96.0229111, -131.1743164, 101.1184311, -225.5941315, 227.1972198
5: -111.2250061, 86.6910477, -117.1874924, 91.3618088, -202.5867920, 203.8785400
6: -106.9501724, 103.4762726, -112.6592102, 109.0083008, -215.9584656, 216.1354675
7: -116.0129623, 97.8613815, -122.2553329, 103.1258469, -219.1388092, 220.1167145
8: -141.5624695, 98.1365051, -149.0306244, 103.2023773, -244.7648315, 247.1671295
9: -106.0678329, 105.3293381, -111.7258606, 110.8950806, -216.9629211, 217.0552063

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5882390, upper bound: 202.5880767
time: 9.18 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5887908, upper bound: 202.5886302
time: 8.51 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -118.3846664, 94.7136688, -120.1966248, 96.1321716, -214.5168152, 214.9102783
1: -100.7750931, 83.8949890, -102.3225784, 85.2027359, -185.9778290, 186.2175598
2: -131.1114044, 85.1534424, -133.1334991, 86.4791031, -217.5905151, 218.2869415
3: -138.0594940, 73.6407013, -140.2410431, 74.7718124, -212.8312988, 213.8817444
4: -127.3811951, 98.2165833, -129.3656464, 99.7404175, -227.1216125, 227.5822296
5: -113.8499680, 88.6805344, -115.5757065, 90.0992813, -203.9492493, 204.2562256
6: -109.4602432, 105.8912735, -111.1170425, 107.5233231, -216.9835663, 217.0083160
7: -118.7090530, 100.1440582, -120.5637360, 101.7116318, -220.4206848, 220.7077942
8: -144.9075623, 100.3887634, -147.0272522, 101.8461609, -246.7537231, 247.4160156
9: -108.5225067, 107.7727509, -110.1938324, 109.3817062, -217.9042053, 217.9665833

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5876514, upper bound: 202.5876202
time: 9.24 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5881540, upper bound: 202.5881540
time: 8.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.89 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5880708, upper bound: 202.5880256
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5877189, upper bound: 202.5875967
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5883775, upper bound: 202.5883351
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5880191, upper bound: 202.5879419
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5884304, upper bound: 202.5884187
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5879129, upper bound: 202.5877539
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5887699, upper bound: 202.5888090
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5882699, upper bound: 202.5881462
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5902593, upper bound: 202.5901214
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5902526, upper bound: 202.5901034
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5897519, upper bound: 202.5896567
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5897466, upper bound: 202.5896518
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5882390, upper bound: 202.5880767
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5887908, upper bound: 202.5886302
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5876514, upper bound: 202.5876202
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -202.5881540, upper bound: 202.5881540

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -114.2022247, 91.3458710, -111.8112411, 89.4532700, -203.6554871, 203.1570892
1: -97.2056427, 80.9564209, -95.2692871, 79.2891998, -176.4948425, 176.2257080
2: -126.5015106, 82.2136536, -123.8554611, 80.4831924, -206.9846954, 206.0691223
3: -133.2328033, 71.0905838, -130.3039246, 69.5876770, -202.8204803, 201.3945007
4: -122.9378128, 94.8039398, -120.3449631, 92.8869629, -215.8247681, 215.1488953
5: -109.8206635, 85.6149216, -107.4539032, 83.8354187, -193.6560822, 193.0688171
6: -105.6019897, 102.1906357, -103.3757553, 100.0974350, -205.6994019, 205.5663757
7: -114.5780106, 96.6127930, -112.1500626, 94.6217575, -209.1997681, 208.7628174
8: -139.7138977, 96.9142227, -136.8852692, 94.9936600, -234.7075500, 233.7994995
9: -104.7159424, 104.0377121, -102.5822372, 101.8670349, -206.5829773, 206.6199493

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5870520, upper bound: 202.5868056
time: 7.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5870628, upper bound: 202.5867958
time: 8.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -112.5793915, 90.0662231, -114.5318146, 91.6449432, -204.2243195, 204.5980225
1: -95.8485489, 79.8118362, -97.5564423, 81.1863480, -177.0348816, 177.3682861
2: -124.7097931, 81.0411758, -126.8377686, 82.3773499, -207.0871429, 207.8789215
3: -131.3034058, 70.0893478, -133.4800415, 71.2413406, -202.5447388, 203.5693970
4: -121.1803589, 93.4669037, -123.2423630, 95.0723495, -216.2527161, 216.7092438
5: -108.2575836, 84.3916397, -110.0734863, 85.8233109, -194.0809021, 194.4651184
6: -104.1045074, 100.7501984, -105.8785934, 102.5009003, -206.6054077, 206.6287842
7: -112.9358749, 95.2423019, -114.8418121, 96.9026489, -209.8385162, 210.0840912
8: -137.7690887, 95.5972366, -140.2164001, 97.2339859, -235.0030518, 235.8136292
9: -103.2269135, 102.5716095, -105.0320053, 104.3063278, -207.5332336, 207.6035919

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5866966, upper bound: 202.5864816
time: 8.46 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5867327, upper bound: 202.5864724
time: 7.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -116.9299850, 93.5196915, -111.8519287, 89.4851456, -206.4151154, 205.3716125
1: -99.4948578, 82.8824997, -95.3028564, 79.3171997, -178.8120270, 178.1853638
2: -129.5079498, 84.1478958, -123.8996735, 80.5083466, -210.0162811, 208.0475769
3: -136.4444427, 72.7751770, -130.3509979, 69.6155930, -206.0600281, 203.1261597
4: -125.8796310, 97.0425949, -120.3885193, 92.9190292, -218.7986603, 217.4310760
5: -112.4463348, 87.6578674, -107.4906540, 83.8684692, -196.3148041, 195.1485291
6: -108.1075287, 104.6064072, -103.4105301, 100.1338348, -208.2413635, 208.0169373
7: -117.2981567, 98.9033813, -112.1889572, 94.6542130, -211.9523621, 211.0923462
8: -143.0060883, 99.1509094, -136.9315033, 95.0276184, -238.0337067, 236.0824127
9: -107.1974030, 106.4745102, -102.6204300, 101.9028397, -209.1002502, 209.0949097

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5871397, upper bound: 202.5869616
time: 8.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5871527, upper bound: 202.5869523
time: 8.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -115.2816620, 92.2195740, -114.5822983, 91.6843643, -206.9660187, 206.8018799
1: -98.1161957, 81.7201767, -97.5977707, 81.2215347, -179.3377380, 179.3179321
2: -127.6882629, 82.9581375, -126.8928223, 82.4107132, -210.0989380, 209.8509521
3: -134.4842377, 71.7592773, -133.5389099, 71.2754440, -205.7596741, 205.2981873
4: -124.0960999, 95.6841965, -123.2967834, 95.1123734, -219.2084656, 218.9809875
5: -110.8582001, 86.4160156, -110.1199265, 85.8642426, -196.7224274, 196.5359497
6: -106.5874863, 103.1433868, -105.9224854, 102.5458298, -209.1333160, 209.0658722
7: -115.6313629, 97.5117722, -114.8906937, 96.9439697, -212.5753174, 212.4024353
8: -141.0310822, 97.8140717, -140.2742920, 97.2759018, -238.3069763, 238.0883636
9: -105.6866302, 104.9859695, -105.0792236, 104.3506622, -210.0372925, 210.0651855

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5867787, upper bound: 202.5866303
time: 9.20 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5867986, upper bound: 202.5866171
time: 8.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -113.6987152, 90.9527817, -115.3965530, 92.3193970, -206.0181122, 206.3493347
1: -96.7849655, 80.6017151, -98.3071365, 81.8148117, -178.5997772, 178.9088440
2: -125.9522324, 81.8591690, -127.8393784, 83.0450745, -208.9972992, 209.6985168
3: -132.6455994, 70.7815094, -134.5485077, 71.7882156, -204.4338074, 205.3300018
4: -122.3884735, 94.3846207, -124.2037964, 95.8163071, -218.2047424, 218.5884094
5: -109.3411713, 85.2389908, -110.9280930, 86.4945526, -195.8357239, 196.1670837
6: -105.1399765, 101.7452011, -106.6918488, 103.2819672, -208.4219360, 208.4370422
7: -114.0695267, 96.1978607, -115.7434235, 97.6622009, -211.7317200, 211.9412842
8: -139.1110535, 96.5009766, -141.2669678, 97.9377670, -237.0488129, 237.7679443
9: -104.2485733, 103.5725632, -105.8306198, 105.0667953, -209.3153534, 209.4031830

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5874615, upper bound: 202.5874155
time: 9.11 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5874958, upper bound: 202.5874148
time: 8.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -112.0558243, 89.6564026, -117.8865891, 94.3293076, -206.3851318, 207.5429840
1: -95.4103622, 79.4423065, -100.3994141, 83.5510178, -178.9613800, 179.8417053
2: -124.1376953, 80.6714630, -130.5732574, 84.7718048, -208.9095001, 211.2447205
3: -130.6925659, 69.7664948, -137.4492188, 73.2963104, -203.9888458, 207.2157135
4: -120.6096039, 93.0309448, -126.8534546, 97.8159180, -218.4255219, 219.8843842
5: -107.7579346, 84.0008850, -113.3191757, 88.3083267, -196.0662537, 197.3200531
6: -103.6232224, 100.2861710, -108.9818039, 105.4839630, -209.1071625, 209.2679596
7: -112.4061508, 94.8095551, -118.2036514, 99.7452927, -212.1514282, 213.0131836
8: -137.1417236, 95.1678696, -144.3256989, 99.9932785, -237.1349792, 239.4935455
9: -102.7416763, 102.0870972, -108.0693817, 107.2944717, -210.0361481, 210.1564636

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 158

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5869671, upper bound: 202.5868121
time: 9.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5870015, upper bound: 202.5868146
time: 7.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -116.4084473, 93.1111526, -115.4491882, 92.3603134, -208.7687531, 208.5603180
1: -99.0590668, 82.5155258, -98.3510590, 81.8515930, -180.9106140, 180.8665771
2: -128.9392090, 83.7810516, -127.8965836, 83.0794296, -212.0186462, 211.6776276
3: -135.8357391, 72.4560699, -134.6087036, 71.8246994, -207.6604309, 207.0647736
4: -125.3112946, 96.6085663, -124.2601776, 95.8576202, -221.1688690, 220.8686829
5: -111.9494781, 87.2683105, -110.9766388, 86.5373230, -198.4868011, 198.2449493
6: -107.6294785, 104.1454468, -106.7375412, 103.3289185, -210.9584045, 210.8829651
7: -116.7718353, 98.4728928, -115.7939682, 97.7045746, -214.4763947, 214.2668610
8: -142.3817749, 98.7232437, -141.3269501, 97.9818802, -240.3636475, 240.0502014
9: -106.7136383, 105.9935074, -105.8790970, 105.1136246, -211.8272705, 211.8726044

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5875757, upper bound: 202.5875765
time: 9.17 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5875852, upper bound: 202.5875676
time: 9.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -114.7416229, 91.7953720, -117.9527054, 94.3805084, -209.1221008, 209.7480774
1: -97.6643295, 81.3394165, -100.4544067, 83.5973740, -181.2617035, 181.7938232
2: -127.0986176, 82.5772400, -130.6451416, 84.8163147, -211.9149323, 213.2223816
3: -133.8539276, 71.4277344, -137.5257874, 73.3412933, -207.1952209, 208.9535217
4: -123.5075989, 95.2349319, -126.9247665, 97.8685837, -221.3761902, 222.1596985
5: -110.3425751, 86.0126114, -113.3807144, 88.3614044, -198.7039795, 199.3933105
6: -106.0917740, 102.6650238, -109.0404892, 105.5430832, -211.6348572, 211.7055054
7: -115.0856476, 97.0647659, -118.2683182, 99.7995224, -214.8851624, 215.3330841
8: -140.3842468, 97.3710709, -144.4019012, 100.0489731, -240.4331818, 241.7729797
9: -105.1860733, 104.4872055, -108.1308517, 107.3535004, -212.5395813, 212.6180573

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5870733, upper bound: 202.5869715
time: 8.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5870837, upper bound: 202.5869627
time: 9.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -113.2944489, 90.5999832, -109.7438660, 87.7145996, -201.0090485, 200.3437958
1: -96.5206680, 80.3382950, -93.5939484, 77.8781433, -174.3988037, 173.9322510
2: -125.5053101, 81.5552063, -121.5913696, 78.9977417, -204.5030518, 203.1465759
3: -132.1198883, 70.5406113, -128.0454559, 68.3926086, -200.5124817, 198.5860596
4: -121.9515610, 94.1042328, -118.1614380, 91.2248535, -213.1763458, 212.2656708
5: -108.9392776, 84.9236298, -105.4742050, 82.3151398, -191.2544098, 190.3978271
6: -104.7620163, 101.3997650, -101.4445343, 98.3148651, -203.0768433, 202.8442841
7: -113.6738434, 95.8841858, -110.1837082, 92.9649200, -206.6387329, 206.0679016
8: -138.6981506, 96.1197815, -134.3182526, 93.0118484, -231.7099915, 230.4380341
9: -103.9522476, 103.2432785, -100.7790375, 100.1021347, -204.0543823, 204.0223083

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5870179, upper bound: 202.5868014
time: 9.28 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5872186, upper bound: 202.5870240
time: 7.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -113.0228729, 90.3784561, -114.5318069, 91.5126266, -204.5354919, 204.9102631
1: -96.2919769, 80.1437836, -97.6067429, 81.2285080, -177.5204773, 177.7505035
2: -125.1936264, 81.3588715, -126.8388062, 82.4150543, -207.6086578, 208.1976776
3: -131.7939148, 70.3762970, -133.6106262, 71.3213730, -203.1152954, 203.9869232
4: -121.6536713, 93.8768539, -123.3254089, 95.1298447, -216.7835083, 217.2022552
5: -108.6782455, 84.7223816, -110.0763702, 85.8890915, -194.5673370, 194.7987518
6: -104.5074234, 101.1567764, -105.8467789, 102.5449753, -207.0523987, 207.0035553
7: -113.3936768, 95.6482697, -114.9612656, 96.9669571, -210.3606262, 210.6094818
8: -138.3671417, 95.9013138, -140.1112518, 96.9855118, -235.3526611, 236.0125732
9: -103.7046661, 102.9963303, -105.1607590, 104.4068451, -208.1114960, 208.1570587

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5869858, upper bound: 202.5867911
time: 9.50 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5871802, upper bound: 202.5870085
time: 8.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -116.0845871, 92.8519592, -108.0949249, 86.4130783, -202.4976654, 200.9468689
1: -98.8677750, 82.2866287, -92.2149658, 76.7146835, -175.5824280, 174.5015869
2: -128.5706024, 83.4962158, -119.7701721, 77.8050537, -206.3756561, 203.2663422
3: -135.3804779, 72.2408295, -126.0833817, 67.3767700, -202.7572479, 198.3242035
4: -124.9233398, 96.3487930, -116.3752747, 89.8668900, -214.7902222, 212.7240601
5: -111.6258316, 86.9596176, -103.8867035, 81.0708008, -192.6966095, 190.8463135
6: -107.3281326, 103.8699875, -99.9226608, 96.8506699, -204.1788025, 203.7926483
7: -116.4334412, 98.2215118, -108.5146255, 91.5716629, -208.0050964, 206.7361450
8: -142.1192169, 98.4216385, -132.3422089, 91.6734543, -233.7926483, 230.7638550
9: -106.4618759, 105.7454910, -99.2669296, 98.6133499, -205.0752258, 205.0124207

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5865306, upper bound: 202.5864123
time: 9.48 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5867007, upper bound: 202.5866061
time: 8.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -115.8704987, 92.6767273, -112.9541779, 90.2700043, -206.1404724, 205.6309052
1: -98.6837921, 82.1319046, -96.2849884, 80.1148071, -178.7985382, 178.4169006
2: -128.3216248, 83.3412552, -125.0970078, 81.2754135, -209.5970306, 208.4382629
3: -135.1235046, 72.1104965, -131.7344818, 70.3477020, -205.4712067, 203.8449707
4: -124.6874542, 96.1673508, -121.6170959, 93.8298798, -218.5173340, 217.7844238
5: -111.4204254, 86.8007812, -108.5574951, 84.6983643, -196.1187744, 195.3582764
6: -107.1255112, 103.6767349, -104.3914795, 101.1436768, -208.2691345, 208.0682068
7: -116.2111282, 98.0321655, -113.3641205, 95.6327667, -211.8438873, 211.3962708
8: -141.8533325, 98.2492905, -138.2194824, 95.7062302, -237.5595551, 236.4687805
9: -106.2647705, 105.5486984, -103.7128220, 102.9819031, -209.2466125, 209.2615204

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5865049, upper bound: 202.5864000
time: 9.26 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5866569, upper bound: 202.5865859
time: 9.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -114.1881790, 91.3529739, -117.7935944, 94.2330322, -208.4212036, 209.1465454
1: -97.2452850, 80.9543610, -100.2937546, 83.5082703, -180.7535553, 181.2480621
2: -126.5031204, 82.2135468, -130.5049438, 84.7909622, -211.2940826, 212.7184906
3: -133.1394348, 71.0621033, -137.4226227, 73.2636261, -206.4030457, 208.4847260
4: -122.9086227, 94.8201218, -126.8255157, 97.7782822, -220.6869049, 221.6456299
5: -109.8059387, 85.5957489, -113.2474518, 88.3208084, -198.1267395, 198.8432007
6: -105.5965652, 102.1828690, -108.9029465, 105.4215851, -211.0181122, 211.0858154
7: -114.5497131, 96.6299057, -118.1951523, 99.7063522, -214.2560577, 214.8250580
8: -139.7972412, 96.9422379, -144.1308441, 99.8944321, -239.6916809, 241.0730896
9: -104.7361069, 104.0114365, -108.0322037, 107.2376938, -211.9737701, 212.0436401

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5872697, upper bound: 202.5872230
time: 8.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5872724, upper bound: 202.5872136
time: 8.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -114.2848969, 91.4286118, -121.0558167, 96.8337097, -211.1186066, 212.4844360
1: -97.3269958, 81.0222321, -103.0338593, 85.8115616, -183.1385498, 184.0560760
2: -126.6093369, 82.2794266, -134.1057587, 87.1011810, -213.7105103, 216.3851929
3: -133.2522888, 71.1252975, -141.2632599, 75.2766418, -208.5288849, 212.3885498
4: -123.0117188, 94.8984070, -130.3417358, 100.4559708, -223.4676819, 225.2401428
5: -109.8972473, 85.6710739, -116.3938217, 90.7617722, -200.6590118, 202.0648956
6: -105.6835403, 102.2685852, -111.8981552, 108.3113785, -213.9949188, 214.1667480
7: -114.6451111, 96.7097473, -121.4498901, 102.4443436, -217.0894012, 218.1596222
8: -139.9108887, 97.0221024, -148.0662994, 102.5698166, -242.4806824, 245.0883942
9: -104.8249054, 104.0979614, -111.0005035, 110.1534729, -214.9783630, 215.0984650

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5875182, upper bound: 202.5874598
time: 8.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5874936, upper bound: 202.5874434
time: 9.24 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -116.9121552, 93.5509033, -116.1155319, 92.9080048, -209.8201599, 209.6664429
1: -99.5354691, 82.8560257, -98.8876114, 82.3241577, -181.8596191, 181.7436218
2: -129.4948730, 84.1066513, -128.6520538, 83.5773926, -213.0722656, 212.7586975
3: -136.3215332, 72.7191315, -135.4245148, 72.2265091, -208.5480347, 208.1436157
4: -125.8094406, 97.0101700, -125.0101318, 96.3950424, -222.2044830, 222.0202942
5: -112.4272308, 87.5817642, -111.6297150, 87.0536499, -199.4808655, 199.2114563
6: -108.1027756, 104.5940933, -107.3547058, 103.9310074, -212.0337830, 211.9487762
7: -117.2416229, 98.9092026, -116.4974060, 98.2867508, -215.5283661, 215.4066162
8: -143.1373901, 99.1908875, -142.1198883, 98.5327301, -241.6701202, 241.3107605
9: -107.1870804, 106.4512634, -106.4943542, 105.7185287, -212.9056091, 212.9456177

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5867115, upper bound: 202.5867673
time: 9.10 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5867289, upper bound: 202.5867557
time: 9.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -117.0048599, 93.6229172, -119.2797775, 95.4274826, -212.4322968, 212.9026947
1: -99.6131821, 82.9208832, -101.5440903, 84.5560532, -184.1691742, 184.4649353
2: -129.5956726, 84.1699295, -132.1423187, 85.8164825, -215.4121552, 216.3122559
3: -136.4294434, 72.7799683, -139.1468353, 74.1829147, -210.6123505, 211.9268036
4: -125.9089737, 97.0852432, -128.4230194, 98.9892654, -224.8982391, 225.5082397
5: -112.5144119, 87.6550140, -114.6774979, 89.4205017, -201.9349060, 202.3325043
6: -108.1861649, 104.6762924, -110.2615662, 106.7318573, -214.9180145, 214.9378357
7: -117.3332443, 98.9856873, -119.6519241, 100.9390030, -218.2722473, 218.6376038
8: -143.2459412, 99.2675323, -145.9355469, 101.1281204, -244.3740387, 245.2030640
9: -107.2728882, 106.5344925, -109.3747940, 108.5432129, -215.8161011, 215.9092712

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 97

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5869317, upper bound: 202.5869615
time: 8.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5869480, upper bound: 202.5869480
time: 9.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.40 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5870520, upper bound: 202.5868056
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5870628, upper bound: 202.5867958
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5866966, upper bound: 202.5864816
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5867327, upper bound: 202.5864724
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5871397, upper bound: 202.5869616
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5871527, upper bound: 202.5869523
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5867787, upper bound: 202.5866303
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5867986, upper bound: 202.5866171
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5874615, upper bound: 202.5874155
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5874958, upper bound: 202.5874148
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5869671, upper bound: 202.5868121
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5870015, upper bound: 202.5868146
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5875757, upper bound: 202.5875765
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5875852, upper bound: 202.5875676
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5870733, upper bound: 202.5869715
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5870837, upper bound: 202.5869627
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5870179, upper bound: 202.5868014
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5872186, upper bound: 202.5870240
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5869858, upper bound: 202.5867911
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5871802, upper bound: 202.5870085
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5865306, upper bound: 202.5864123
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5867007, upper bound: 202.5866061
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5865049, upper bound: 202.5864000
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5866569, upper bound: 202.5865859
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5872697, upper bound: 202.5872230
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5872724, upper bound: 202.5872136
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5875182, upper bound: 202.5874598
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5874936, upper bound: 202.5874434
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5867115, upper bound: 202.5867673
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5867289, upper bound: 202.5867557
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5869317, upper bound: 202.5869615
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -202.5869480, upper bound: 202.5869480

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -105.6307907, 84.4460754, -109.0017471, 87.1916351, -192.8223877, 193.4478149
1: -90.0891495, 74.9538803, -92.9356461, 77.3215561, -167.4107056, 167.8895111
2: -117.0619736, 76.0624084, -120.7603073, 78.4677277, -195.5296936, 196.8227234
3: -123.2493744, 65.8592453, -127.0312119, 67.8730927, -191.1224670, 192.8904419
4: -113.7471924, 87.8149643, -117.3319931, 90.5951767, -204.3423767, 205.1469574
5: -101.5473404, 79.2038727, -104.7426529, 81.7340469, -183.2813721, 183.9465179
6: -97.6714325, 94.6520920, -100.7765198, 97.6260300, -195.2974548, 195.4286194
7: -106.0695724, 89.4644852, -109.3606949, 92.2787247, -198.3482971, 198.8251648
8: -129.3439178, 89.6418610, -133.4846802, 92.6088104, -221.9527283, 223.1265411
9: -96.9908218, 96.4146652, -100.0494003, 99.3682938, -196.3591156, 196.4640656

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5764110, upper bound: 202.5758786
time: 7.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5798000, upper bound: 202.5794440
time: 8.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 17.46 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 17.46
Output dim: 1, lower bound: -202.5764110, upper bound: 202.5758786
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 17.46
Output dim: 1, lower bound: -202.5798000, upper bound: 202.5794440
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5870628, upper bound: 202.5867958
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5866966, upper bound: 202.5864816
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5867327, upper bound: 202.5864724
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5871397, upper bound: 202.5869616
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5871527, upper bound: 202.5869523
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5867787, upper bound: 202.5866303
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5867986, upper bound: 202.5866171
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5874615, upper bound: 202.5874155
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5874958, upper bound: 202.5874148
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5869671, upper bound: 202.5868121
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5870015, upper bound: 202.5868146
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5875757, upper bound: 202.5875765
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5875852, upper bound: 202.5875676
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5870733, upper bound: 202.5869715
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5870837, upper bound: 202.5869627
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5870179, upper bound: 202.5868014
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5872186, upper bound: 202.5870240
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5869858, upper bound: 202.5867911
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5871802, upper bound: 202.5870085
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5865306, upper bound: 202.5864123
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5867007, upper bound: 202.5866061
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5865049, upper bound: 202.5864000
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5866569, upper bound: 202.5865859
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5872697, upper bound: 202.5872230
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5872724, upper bound: 202.5872136
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5875182, upper bound: 202.5874598
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5874936, upper bound: 202.5874434
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5867115, upper bound: 202.5867673
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5867289, upper bound: 202.5867557
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5869317, upper bound: 202.5869615
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.46
Output dim: 1, lower bound: -202.5869480, upper bound: 202.5869480
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=203.5233612060547
rel_dist={1: [-202.60871310808878, 202.60871310808875]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6041051, upper bound: 202.6040783
time: 9.55 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6040999, upper bound: 202.6040999
time: 9.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.61
Output dim: 1, lower bound: -202.6041051, upper bound: 202.6040783
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.61
Output dim: 1, lower bound: -202.6040999, upper bound: 202.6040999

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -123.0395737, 98.3348694, -126.6315308, 101.1876678, -224.2272339, 224.9664001
1: -104.6226501, 87.1827240, -107.6576080, 89.7174377, -194.3400879, 194.8402863
2: -136.2332764, 88.5248413, -140.2068634, 91.1035614, -227.3368378, 228.7317047
3: -143.6921082, 76.5960464, -147.9042358, 78.8033981, -222.4955139, 224.5002747
4: -132.4142151, 102.0421982, -136.2878265, 105.0055237, -237.4197235, 238.3300171
5: -118.3712463, 92.2120056, -121.8298645, 94.9008331, -213.2720795, 214.0418701
6: -113.7487183, 109.9841461, -117.0497894, 113.1722336, -226.9209595, 227.0339355
7: -123.4394455, 104.0390549, -127.0597839, 107.0793839, -230.5187988, 231.0988312
8: -150.3356781, 104.1154709, -154.6888123, 107.0769272, -257.4125366, 258.8042908
9: -112.7398911, 111.9838409, -116.0285034, 115.2059250, -227.9458160, 228.0123444

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5978795, upper bound: 202.5978206
time: 10.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5983707, upper bound: 202.5983424
time: 10.08 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.7521286, 100.5080948, -127.0627594, 101.5400009, -227.2921295, 227.5708618
1: -106.9180679, 89.1028976, -108.0283508, 90.0258789, -196.9439392, 197.1312408
2: -139.2514191, 90.4931870, -140.6967926, 91.4238739, -230.6752930, 231.1899719
3: -146.8589630, 78.2464371, -148.4018402, 79.0627670, -225.9217224, 226.6482849
4: -135.3592224, 104.2918320, -136.7671967, 105.3725739, -240.7317810, 241.0590210
5: -120.9924850, 94.2561188, -122.2443237, 95.2349625, -216.2274475, 216.5004272
6: -116.2524948, 112.4022141, -117.4522476, 113.5646210, -229.8171082, 229.8544464
7: -126.1867523, 106.3574982, -127.5123367, 107.4626007, -233.6493530, 233.8698425
8: -153.6653137, 106.3638687, -155.2318115, 107.4378052, -261.1030273, 261.5956726
9: -115.2547913, 114.4284286, -116.4510117, 115.6080093, -230.8627930, 230.8794403

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5978830, upper bound: 202.5978482
time: 10.16 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5984140, upper bound: 202.5984140
time: 9.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.29 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.29
Output dim: 1, lower bound: -202.5978795, upper bound: 202.5978206
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.29
Output dim: 1, lower bound: -202.5983707, upper bound: 202.5983424
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.29
Output dim: 1, lower bound: -202.5978830, upper bound: 202.5978482
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.29
Output dim: 1, lower bound: -202.5984140, upper bound: 202.5984140

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -116.6660385, 93.2973022, -117.9316406, 94.3103561, -210.9763947, 211.2289429
1: -99.2758560, 82.7068176, -100.3579102, 83.6081390, -182.8839722, 183.0647278
2: -129.1926422, 83.9646225, -130.5960083, 84.8783188, -214.0709381, 214.5606079
3: -136.1532135, 72.6288147, -137.6134796, 73.3897476, -209.5429535, 210.2422943
4: -125.5584412, 96.8186874, -126.9292603, 97.8745193, -223.4329529, 223.7479401
5: -112.1973267, 87.4768066, -113.4010544, 88.4372864, -200.6345978, 200.8778687
6: -107.8704529, 104.3635406, -109.0247726, 105.4995651, -213.3700104, 213.3883057
7: -117.0094986, 98.6859436, -118.2833710, 99.7711182, -216.7806091, 216.9693146
8: -142.6542206, 98.8930740, -144.2031097, 99.9485931, -242.6027985, 243.0961914
9: -106.9464798, 106.2407379, -108.1201706, 107.3652039, -214.3116760, 214.3609009

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5907162, upper bound: 202.5906498
time: 9.48 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5905473, upper bound: 202.5905138
time: 9.55 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -116.4623184, 93.1439514, -121.4377823, 97.1127243, -213.5750427, 214.5816956
1: -99.1075134, 82.5622025, -103.3289948, 86.0775070, -185.1850128, 185.8911896
2: -128.9816437, 83.8294678, -134.4911957, 87.3831558, -216.3648071, 218.3206329
3: -135.9300079, 72.5113907, -141.7593079, 75.5418701, -211.4718628, 214.2706909
4: -125.3244858, 96.6383667, -130.7022247, 100.7392654, -226.0637512, 227.3405914
5: -112.0158844, 87.3220596, -116.7961731, 91.0383530, -203.0542297, 204.1182251
6: -107.6870880, 104.1808701, -112.2668915, 108.6136398, -216.3007202, 216.4477386
7: -116.8029099, 98.5304413, -121.7966919, 102.7432022, -219.5460968, 220.3271332
8: -142.4144745, 98.7226715, -148.4903412, 102.8301239, -245.2445831, 247.2130127
9: -106.7429047, 106.0342026, -111.2978058, 110.4919968, -217.2348938, 217.3320007

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5909270, upper bound: 202.5908175
time: 10.49 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5907538, upper bound: 202.5906700
time: 10.34 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -119.3564987, 95.4533920, -118.3178253, 94.6279602, -213.9844666, 213.7712097
1: -101.5503693, 84.6114655, -100.6899109, 83.8847885, -185.4351501, 185.3013458
2: -132.1841736, 85.9164200, -131.0346985, 85.1655426, -217.3497009, 216.9511108
3: -139.2944031, 74.2662125, -138.0589752, 73.6219482, -212.9163513, 212.3251801
4: -128.4786987, 99.0505066, -127.3601532, 98.2049561, -226.6836395, 226.4106598
5: -114.7972946, 89.5059433, -113.7731323, 88.7385483, -203.5358124, 203.2790833
6: -110.3532639, 106.7596588, -109.3861694, 105.8509140, -216.2041779, 216.1458282
7: -119.7319183, 100.9840240, -118.6877518, 100.1153259, -219.8472290, 219.6717682
8: -145.9562073, 101.1245499, -144.6908875, 100.2741318, -246.2303314, 245.8154297
9: -109.4408035, 108.6649094, -108.5019226, 107.7267532, -217.1675262, 217.1668396

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5906851, upper bound: 202.5906658
time: 8.89 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5904791, upper bound: 202.5905141
time: 10.97 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -119.1593475, 95.3035126, -121.8956680, 97.4850845, -216.6444244, 217.1991730
1: -101.3904724, 84.4707642, -103.7230377, 86.4041367, -187.7946014, 188.1938019
2: -131.9827423, 85.7852631, -135.0111084, 87.7209396, -219.7036743, 220.7963715
3: -139.0780334, 74.1513824, -142.2893524, 75.8173141, -214.8953247, 216.4407349
4: -128.2525940, 98.8758316, -131.2114868, 101.1280823, -229.3806610, 230.0873108
5: -114.6211014, 89.3558350, -117.2366638, 91.3928146, -206.0139160, 206.5924988
6: -110.1760788, 106.5843811, -112.6940231, 109.0301895, -219.2062683, 219.2784119
7: -119.5337601, 100.8362350, -122.2751617, 103.1492157, -222.6829529, 223.1113586
8: -145.7235565, 100.9582672, -149.0644989, 103.2127838, -248.9363403, 250.0227661
9: -109.2422791, 108.4663925, -111.7437820, 110.9197693, -220.1620483, 220.2101593

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5908902, upper bound: 202.5908346
time: 9.65 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5906548, upper bound: 202.5906548
time: 12.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.87 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 1, lower bound: -202.5907162, upper bound: 202.5906498
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 1, lower bound: -202.5905473, upper bound: 202.5905138
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 1, lower bound: -202.5909270, upper bound: 202.5908175
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 1, lower bound: -202.5907538, upper bound: 202.5906700
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 1, lower bound: -202.5906851, upper bound: 202.5906658
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 1, lower bound: -202.5904791, upper bound: 202.5905141
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 1, lower bound: -202.5908902, upper bound: 202.5908346
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.87
Output dim: 1, lower bound: -202.5906548, upper bound: 202.5906548

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -110.0171890, 88.0239639, -113.1603928, 90.5254822, -200.5426636, 201.1843567
1: -93.7297668, 78.0155792, -96.3771057, 80.2414780, -173.9712219, 174.3926849
2: -121.8566360, 79.1887283, -125.3307877, 81.4511261, -203.3077698, 204.5195007
3: -128.2317963, 68.5026779, -131.9285431, 70.4301605, -198.6619568, 200.4312134
4: -118.3870468, 91.3845520, -121.7843246, 93.9736786, -212.3607178, 213.1688690
5: -105.7626190, 82.4732056, -108.7822800, 84.8469696, -190.6095886, 191.2554779
6: -101.7419662, 98.4798660, -104.6277084, 101.2764206, -203.0183868, 203.1075592
7: -110.3123474, 93.0751419, -113.4771881, 95.7440262, -206.0563660, 206.5522766
8: -134.7046051, 93.4934387, -138.4976196, 96.0735550, -230.7781677, 231.9910583
9: -100.9019318, 100.2583160, -103.7833481, 103.0714722, -203.9733887, 204.0416565

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5868585, upper bound: 202.5867695
time: 9.98 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5873259, upper bound: 202.5872518
time: 10.11 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -112.8227768, 90.2828979, -111.4781494, 89.2030792, -202.0258484, 201.7610474
1: -96.0914307, 79.9743423, -94.9668655, 79.0536270, -175.1450500, 174.9412079
2: -124.9340820, 81.1452560, -123.4725113, 80.2294235, -205.1635132, 204.6177673
3: -131.5093231, 70.2093277, -129.9291840, 69.3955460, -200.9048767, 200.1385040
4: -121.3802109, 93.6401825, -119.9565887, 92.5780182, -213.9581909, 213.5967712
5: -108.4629440, 84.5283432, -107.1680374, 83.5736771, -192.0366211, 191.6963806
6: -104.3262024, 100.9615402, -103.0737915, 99.7792969, -204.1054535, 204.0353241
7: -113.0933304, 95.4317017, -111.7681961, 94.3216705, -207.4149933, 207.1998901
8: -138.1420288, 95.8062057, -136.4795990, 94.7086716, -232.8506927, 232.2857666
9: -103.4338684, 102.7742767, -102.2286682, 101.5452042, -204.9790344, 205.0029297

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 158

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5866863, upper bound: 202.5866359
time: 9.52 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5871409, upper bound: 202.5871168
time: 9.71 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -109.8667984, 87.9142761, -116.7180481, 93.3697357, -203.2365265, 204.6322937
1: -93.6072311, 77.9089127, -99.3921814, 82.7474594, -176.3546906, 177.3010254
2: -121.7052383, 79.0920486, -129.2832947, 83.9932022, -205.6984406, 208.3753357
3: -128.0757599, 68.4191284, -136.1390076, 72.6146698, -200.6904297, 204.5581360
4: -118.2100067, 91.2483444, -125.6129913, 96.8808594, -215.0908661, 216.8613129
5: -105.6353455, 82.3579788, -112.2293015, 87.4863205, -193.1216583, 194.5872803
6: -101.6091690, 98.3449173, -107.9186325, 104.4365845, -206.0457306, 206.2635345
7: -110.1590729, 92.9656830, -117.0418777, 98.7604065, -208.9194794, 210.0075378
8: -134.5303192, 93.3658676, -142.8464661, 98.9959869, -233.5263062, 236.2123413
9: -100.7463074, 100.1019974, -107.0070877, 106.2455368, -206.9918518, 207.1090698

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5870383, upper bound: 202.5869327
time: 9.09 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5876415, upper bound: 202.5875323
time: 9.08 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -112.5842361, 90.1007614, -114.9232178, 91.9567108, -204.5409546, 205.0239868
1: -95.8940887, 79.8072128, -97.8850403, 81.4804840, -177.3745728, 177.6922150
2: -124.6843719, 80.9822159, -127.3010864, 82.6896439, -207.3740234, 208.2832947
3: -131.2476349, 70.0738373, -134.0034637, 71.5078735, -202.7555084, 204.0773010
4: -121.1086960, 93.4313202, -123.6644211, 95.3941650, -216.5028534, 217.0957184
5: -108.2452698, 84.3460388, -110.5026169, 86.1287003, -194.3739624, 194.8486328
6: -104.1114883, 100.7503204, -106.2599258, 102.8388748, -206.9503479, 207.0102386
7: -112.8516388, 95.2435532, -115.2207031, 97.2400742, -210.0917053, 210.4642639
8: -137.8586731, 95.6101761, -140.6923523, 97.5419998, -235.4006653, 236.3025208
9: -103.2000656, 102.5342712, -105.3508453, 104.6147842, -207.8148499, 207.8851166

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5868775, upper bound: 202.5868163
time: 9.29 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5874199, upper bound: 202.5873679
time: 11.28 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -112.6691971, 90.1472626, -113.5647049, 90.8559341, -203.5251312, 203.7119293
1: -95.9724350, 79.8930130, -96.7246170, 80.5310059, -176.5034027, 176.6175995
2: -124.8058319, 81.1132278, -125.7894897, 81.7513504, -206.5571594, 206.9027100
3: -131.3282928, 70.1167831, -132.3957977, 70.6731567, -202.0014496, 202.5125732
4: -121.2681274, 93.5828018, -122.2351608, 94.3186874, -215.5868225, 215.8179321
5: -108.3231735, 84.4733276, -109.1714401, 85.1614761, -193.4846344, 193.6447754
6: -104.1919098, 100.8422394, -105.0067139, 101.6443329, -205.8362427, 205.8489532
7: -112.9970474, 95.3412247, -113.9001312, 96.1042862, -209.1013031, 209.2413483
8: -137.9590607, 95.6924591, -139.0066071, 96.4130402, -234.3721008, 234.6990662
9: -103.3629608, 102.6469574, -104.1820068, 103.4489365, -206.8118744, 206.8289337

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5891471, upper bound: 202.5891011
time: 9.83 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5891448, upper bound: 202.5891002
time: 10.95 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -115.4571533, 92.3993530, -111.7999573, 89.4702682, -204.9273834, 204.1993103
1: -98.3160324, 81.8389053, -95.2439804, 79.2857056, -177.6017303, 177.0828857
2: -127.8696747, 83.0555267, -123.8406296, 80.4700775, -208.3397369, 206.8961487
3: -134.5867004, 71.8099518, -130.2998505, 69.5881882, -204.1748962, 202.1098022
4: -124.2376404, 95.8255386, -120.3173828, 92.8557816, -217.0933838, 216.1429138
5: -111.0089874, 86.5079498, -107.4787903, 83.8260117, -194.8349915, 193.9867401
6: -106.7564545, 103.3081589, -103.3764496, 100.0735703, -206.8300171, 206.6845856
7: -115.7559967, 97.6779709, -112.1074524, 94.6107712, -210.3667603, 209.7854309
8: -141.3782959, 97.9897614, -136.8898926, 94.9805298, -236.3588257, 234.8796387
9: -105.8694305, 105.1472092, -102.5506210, 101.8490372, -207.7184753, 207.6977844

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5889762, upper bound: 202.5889466
time: 11.81 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5889759, upper bound: 202.5889450
time: 9.63 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -112.5267563, 90.0427780, -117.1961975, 93.7577133, -206.2844543, 207.2389221
1: -95.8584747, 79.7919769, -99.8026962, 83.0888596, -178.9473267, 179.5946503
2: -124.6655960, 81.0228806, -129.8260803, 84.3465652, -209.0121613, 210.8489685
3: -131.1805115, 70.0370255, -136.6932220, 72.9023666, -204.0828857, 206.7302551
4: -121.1008453, 93.4528809, -126.1440964, 97.2861328, -218.3869781, 219.5969696
5: -108.2012711, 84.3640289, -112.6881485, 87.8559036, -196.0571594, 197.0521698
6: -104.0649872, 100.7153854, -108.3644257, 104.8712082, -208.9361725, 209.0797882
7: -112.8547134, 95.2402725, -117.5421066, 99.1838226, -212.0385437, 212.7823334
8: -137.7919922, 95.5709686, -143.4437561, 99.3951416, -237.1871338, 239.0147247
9: -103.2126541, 102.4995956, -107.4712830, 106.6914597, -209.9041138, 209.9708862

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5870297, upper bound: 202.5869599
time: 10.64 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5876323, upper bound: 202.5875663
time: 12.95 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -115.2263565, 92.2203903, -115.3144379, 92.2762299, -207.5025635, 207.5348206
1: -98.1273499, 81.6753082, -98.2220993, 81.7602844, -179.8876190, 179.8973999
2: -127.6289215, 82.8976135, -127.7468109, 82.9793396, -210.6082611, 210.6444244
3: -134.3309784, 71.6765976, -134.4547119, 71.7415466, -206.0725098, 206.1313019
4: -123.9754715, 95.6222763, -124.1018829, 95.7276688, -219.7031403, 219.7241516
5: -110.7967911, 86.3311157, -110.8793259, 86.4317627, -197.2285156, 197.2104340
6: -106.5484161, 103.1040421, -106.6263351, 103.1960220, -209.7443848, 209.7303772
7: -115.5226593, 97.4982681, -115.6311569, 97.5892715, -213.1119385, 213.1294098
8: -141.1035156, 97.7994766, -141.1866913, 97.8702698, -238.9737854, 238.9861755
9: -105.6430130, 104.9146118, -105.7366562, 104.9818726, -210.6248779, 210.6512604

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5868222, upper bound: 202.5868062
time: 10.89 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5873839, upper bound: 202.5873839
time: 10.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.37 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5868585, upper bound: 202.5867695
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5873259, upper bound: 202.5872518
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5866863, upper bound: 202.5866359
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5871409, upper bound: 202.5871168
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5870383, upper bound: 202.5869327
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5876415, upper bound: 202.5875323
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5868775, upper bound: 202.5868163
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5874199, upper bound: 202.5873679
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5891471, upper bound: 202.5891011
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5891448, upper bound: 202.5891002
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5889762, upper bound: 202.5889466
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5889759, upper bound: 202.5889450
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5870297, upper bound: 202.5869599
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5876323, upper bound: 202.5875663
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5868222, upper bound: 202.5868062
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 1, lower bound: -202.5873839, upper bound: 202.5873839

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -107.0990295, 85.7176743, -109.1936264, 87.3906860, -194.4897156, 194.9112854
1: -91.2740555, 75.9562302, -93.0390472, 77.4428329, -168.7168732, 168.9952698
2: -118.6540833, 77.1123276, -120.9769058, 78.6313019, -197.2853851, 198.0892181
3: -124.7864227, 66.6817169, -127.2463989, 67.9567566, -192.7431793, 193.9280853
4: -115.2732849, 88.9928513, -117.5510941, 90.7223892, -205.9956665, 206.5439453
5: -102.9407883, 80.2947388, -104.9464035, 81.8862915, -184.8270721, 185.2411499
6: -99.0510254, 95.9124146, -100.9713287, 97.7862320, -196.8372498, 196.8837433
7: -107.4068756, 90.6258240, -109.5263596, 92.4150925, -199.8219147, 200.1521606
8: -131.1943817, 91.1233063, -133.7278137, 92.8543854, -224.0487518, 224.8511200
9: -98.2570419, 97.6385422, -100.1874161, 99.5115204, -197.7685547, 197.8259583

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5858482, upper bound: 202.5857822
time: 9.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5858442, upper bound: 202.5857768
time: 9.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -107.3582458, 85.9224701, -112.4271393, 89.9721222, -197.3303680, 198.3496094
1: -91.4911346, 76.1389694, -95.7538071, 79.7248688, -171.2160034, 171.8927765
2: -118.9381561, 77.2911148, -124.5456314, 80.9211426, -199.8592987, 201.8367462
3: -125.0923080, 66.8494949, -131.0533142, 69.9474258, -195.0397339, 197.9028015
4: -115.5512009, 89.2029037, -121.0378647, 93.3763351, -208.9275360, 210.2407532
5: -103.1860809, 80.4942093, -108.0636215, 84.3064728, -187.4925537, 188.5578308
6: -99.2858200, 96.1423111, -103.9390335, 100.6500473, -199.9358673, 200.0813446
7: -107.6616745, 90.8406906, -112.7522583, 95.1290588, -202.7907410, 203.5929565
8: -131.5028076, 91.3378296, -137.6299591, 95.5065994, -227.0093994, 228.9677887
9: -98.4939499, 97.8698730, -103.1300507, 102.4009628, -200.8948822, 200.9999237

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5860553, upper bound: 202.5859729
time: 11.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5860511, upper bound: 202.5859645
time: 10.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -109.8966446, 87.9702759, -107.5005341, 86.0591736, -195.9558105, 195.4708099
1: -93.6287842, 77.9092789, -91.6196518, 76.2472534, -169.8760223, 169.5289307
2: -121.7223206, 79.0618439, -119.1064835, 77.4000015, -199.1223145, 198.1683044
3: -128.0554504, 68.3829803, -125.2345657, 66.9144745, -194.9699097, 193.6175537
4: -118.2568283, 91.2416000, -115.7116699, 89.3178482, -207.5746765, 206.9532318
5: -105.6331177, 82.3422012, -103.3209686, 80.6036758, -186.2367859, 185.6631622
6: -101.6271973, 98.3863983, -99.4070206, 96.2793045, -197.9064941, 197.7934113
7: -110.1790161, 92.9749374, -107.8064194, 90.9824524, -201.1614380, 200.7813568
8: -134.6216431, 93.4278107, -131.6963959, 91.4804611, -226.1021118, 225.1241760
9: -100.7813110, 100.1474380, -98.6229095, 97.9752274, -198.7565155, 198.7703552

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5857150, upper bound: 202.5856688
time: 9.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5857264, upper bound: 202.5856689
time: 11.28 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -110.1739731, 88.1892319, -110.6205750, 88.5475693, -198.7215424, 198.8098145
1: -93.8606110, 78.1056519, -94.2382278, 78.4479370, -172.3085480, 172.3438721
2: -122.0268784, 79.2560730, -122.5481720, 79.6101685, -201.6370544, 201.8042450
3: -128.3822784, 68.5625076, -128.9039917, 68.8402405, -197.2225037, 197.4664764
4: -118.5555344, 91.4662323, -119.0795670, 91.8758545, -210.4313965, 210.5457916
5: -105.8963089, 82.5571213, -106.3258667, 82.9394150, -188.8357239, 188.8829651
6: -101.8791199, 98.6329498, -102.2721176, 99.0407867, -200.9198914, 200.9050446
7: -110.4532242, 93.2063522, -110.9169006, 93.5995255, -204.0527496, 204.1232605
8: -134.9518585, 93.6577911, -135.4611206, 94.0402679, -228.9921112, 229.1189117
9: -101.0363693, 100.3945084, -101.4645538, 100.7599030, -201.7962646, 201.8590698

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5859164, upper bound: 202.5858601
time: 9.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5859204, upper bound: 202.5858547
time: 8.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -106.9286041, 85.5936203, -112.7120209, 90.2052917, -197.1338959, 198.3056030
1: -91.1343536, 75.8352737, -96.0209351, 79.9215317, -171.0558624, 171.8562012
2: -118.4809036, 77.0013733, -124.8857803, 81.1449127, -199.6258087, 201.8871460
3: -124.6066589, 66.5838852, -131.4117126, 70.1162796, -194.7229309, 197.9956055
4: -115.0751190, 88.8410492, -121.3375473, 93.5976868, -208.6727753, 210.1785889
5: -102.7944870, 80.1646729, -108.3555908, 84.4965897, -187.2910767, 188.5202484
6: -98.9001465, 95.7594528, -104.2262573, 100.9110107, -199.8111420, 199.9857025
7: -107.2336044, 90.4997025, -113.0519409, 95.3984604, -202.6320648, 203.5516357
8: -130.9979858, 90.9812546, -138.0304565, 95.7465591, -226.7445374, 229.0117188
9: -98.0834579, 97.4647598, -103.3766251, 102.6515427, -200.7349854, 200.8413849

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5861687, upper bound: 202.5861050
time: 9.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5861730, upper bound: 202.5861051
time: 12.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -107.2167511, 85.8206863, -115.8136597, 92.6778259, -199.8945770, 201.6343384
1: -91.3763733, 76.0386200, -98.6264038, 82.1101837, -173.4865417, 174.6650238
2: -118.7967758, 77.2022171, -128.3083344, 83.3415451, -202.1383057, 205.5105591
3: -124.9457321, 66.7706909, -135.0600739, 72.0323944, -196.9781189, 201.8307648
4: -115.3840332, 89.0743256, -124.6861496, 96.1410370, -211.5250549, 213.7604675
5: -103.0679474, 80.3869247, -111.3437576, 86.8173523, -189.8852997, 191.7306824
6: -99.1616898, 96.0151978, -107.0756683, 103.6572800, -202.8189392, 203.0908661
7: -107.5174408, 90.7389145, -116.1443558, 97.9995651, -205.5169983, 206.8832703
8: -131.3409424, 91.2199860, -141.7728271, 98.2923431, -229.6332703, 232.9927826
9: -98.3460312, 97.7226868, -106.2002029, 105.4197388, -203.7657776, 203.9228821

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5864331, upper bound: 202.5863396
time: 10.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5864283, upper bound: 202.5863306
time: 10.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -109.5652924, 87.7175217, -110.9050980, 88.7826920, -198.3479919, 198.6226044
1: -93.3530121, 77.6759262, -94.5036697, 78.6459427, -171.9989471, 172.1795959
2: -121.3707428, 78.8353043, -122.8895187, 79.8320007, -201.2027435, 201.7248077
3: -127.6833649, 68.1837311, -129.2615814, 69.0014267, -196.6847839, 197.4452972
4: -117.8860397, 90.9583130, -119.3765335, 92.1007004, -209.9867401, 210.3348389
5: -105.3274765, 82.0928726, -106.6172333, 83.1300964, -188.4575806, 188.7101135
6: -101.3281555, 98.0913315, -102.5560303, 99.3025284, -200.6306763, 200.6473694
7: -109.8440628, 92.7107315, -111.2180252, 93.8672791, -203.7113342, 203.9287262
8: -134.2309265, 93.1565704, -135.8615875, 94.2823639, -228.5132599, 229.0181580
9: -100.4627609, 99.8257370, -101.7095718, 101.0094757, -201.4722290, 201.5353088

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5860325, upper bound: 202.5859889
time: 11.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5860392, upper bound: 202.5859866
time: 9.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -109.8788681, 87.9639511, -114.0036011, 91.2513351, -201.1301880, 201.9675293
1: -93.6160278, 77.8978271, -97.1055527, 80.8322525, -174.4482727, 175.0033875
2: -121.7155914, 79.0554276, -126.3078003, 82.0276489, -203.7432404, 205.3631744
3: -128.0535431, 68.3864746, -132.9062958, 70.9163437, -198.9698639, 201.2927551
4: -118.2234192, 91.2132187, -122.7218170, 94.6418228, -212.8652344, 213.9350281
5: -105.6256256, 82.3343124, -109.6013641, 85.4485931, -191.0742188, 191.9356689
6: -101.6143417, 98.3701096, -105.4032135, 102.0456696, -203.6600037, 203.7733154
7: -110.1549911, 92.9723053, -114.3084412, 96.4660110, -206.6210022, 207.2807465
8: -134.6046143, 93.4167786, -139.5999756, 96.8261490, -231.4307556, 233.0167542
9: -100.7497940, 100.1067810, -104.5306931, 103.7754211, -204.5252075, 204.6374817

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5862709, upper bound: 202.5862027
time: 9.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5862685, upper bound: 202.5861958
time: 10.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -106.4428558, 85.1351700, -105.0369186, 83.9916687, -190.4344940, 190.1720886
1: -90.8012390, 75.5324173, -89.6448288, 74.5598679, -165.3611145, 165.1772461
2: -117.9478378, 76.6454926, -116.3995056, 75.6320572, -193.5798950, 193.0449982
3: -124.0747757, 66.3174210, -122.4635773, 65.4691238, -189.5438843, 188.7809601
4: -114.5912781, 88.5043030, -113.0928879, 87.3650665, -201.9563446, 201.5971680
5: -102.3143234, 79.8160629, -100.9408951, 78.7835541, -181.0978699, 180.7569580
6: -98.4325943, 95.3662033, -97.1185379, 94.1468964, -192.5794983, 192.4847412
7: -106.8140564, 90.1478271, -105.4346313, 88.9926147, -195.8066711, 195.5823975
8: -130.4258423, 90.4082794, -128.6908875, 89.1777649, -219.6036072, 219.0991669
9: -97.7487640, 97.1097870, -96.4946899, 95.8675156, -193.6162720, 193.6044617

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5858651, upper bound: 202.5857931
time: 11.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5860542, upper bound: 202.5860058
time: 8.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -106.6483688, 85.2946548, -109.8057556, 87.7745514, -194.4229126, 195.1003723
1: -90.9654007, 75.6687546, -93.6431580, 77.8967056, -168.8621063, 169.3118896
2: -118.1487885, 76.7894287, -121.6266708, 79.0355530, -197.1842957, 198.4161072
3: -124.2960739, 66.4452438, -128.0061951, 68.3868408, -192.6828766, 194.4514465
4: -114.7991104, 88.6593170, -118.2360306, 91.2553558, -206.0544739, 206.8953552
5: -102.5150757, 79.9730225, -105.5251846, 82.3437119, -184.8587952, 185.4981995
6: -98.6165237, 95.5381393, -101.5028381, 98.3609848, -196.9774780, 197.0409851
7: -106.9983215, 90.3000107, -110.1934738, 92.9796982, -199.9780273, 200.4934387
8: -130.6689301, 90.6081390, -134.4617615, 93.1362076, -223.8051453, 225.0699005
9: -97.9299698, 97.2810211, -100.8602371, 100.1561661, -198.0861359, 198.1412354

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5858537, upper bound: 202.5857909
time: 10.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5860410, upper bound: 202.5859952
time: 10.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -109.1834946, 87.3498688, -103.1841202, 82.5347748, -191.7182617, 190.5339966
1: -93.1066132, 77.4452667, -88.0928650, 73.2534256, -166.3600464, 165.5380859
2: -120.9586945, 78.5509186, -114.3531647, 74.2848892, -195.2435913, 192.9040833
3: -127.2759247, 67.9824142, -120.2615814, 64.3324280, -191.6083374, 188.2439575
4: -117.5102386, 90.7075958, -111.0805817, 85.8319397, -203.3421783, 201.7881317
5: -104.9545670, 81.8144455, -99.1641235, 77.3818207, -182.3363953, 180.9785614
6: -100.9532928, 97.7911987, -95.4069824, 92.4990616, -193.4523468, 193.1981659
7: -109.5255127, 92.4448624, -103.5543213, 87.4259567, -196.9514771, 195.9991455
8: -133.7909698, 92.6675262, -126.4700546, 87.6740952, -221.4650574, 219.1375732
9: -100.2128983, 99.5686798, -94.7853394, 94.1907120, -194.4036102, 194.3540192

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5768187, upper bound: 202.5769967
time: 11.01 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5816509, upper bound: 202.5815760
time: 9.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -109.5117569, 87.6073914, -108.0795288, 86.4215698, -195.9333191, 195.6869049
1: -93.3674240, 77.6661377, -92.1917496, 76.6773071, -170.0447388, 169.8578796
2: -121.2938766, 78.7856445, -119.7191544, 77.7824631, -199.0763397, 198.5047913
3: -127.6439362, 68.1822357, -125.9555054, 67.3241501, -194.9680786, 194.1377411
4: -117.8486938, 90.9615555, -116.3610229, 89.8235855, -207.6722717, 207.3225708
5: -105.2729034, 82.0628204, -103.8696136, 81.0363388, -186.3092346, 185.9324341
6: -101.2487717, 98.0685959, -99.9085159, 96.8229218, -198.0716858, 197.9770813
7: -109.8325577, 92.6976547, -108.4387283, 91.5163879, -201.3489227, 201.1363831
8: -134.1742096, 92.9646072, -132.3889923, 91.7369614, -225.9111328, 225.3535767
9: -100.5020065, 99.8471603, -99.2631607, 98.5910034, -199.0930176, 199.1102753

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 97

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5856804, upper bound: 202.5856469
time: 11.50 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5858636, upper bound: 202.5858531
time: 10.21 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -109.5397263, 87.6831589, -113.1188812, 90.5367661, -200.0764923, 200.8020325
1: -93.3436813, 77.6842117, -96.3706970, 80.2127838, -173.5564575, 174.0548706
2: -121.3865433, 78.8984833, -125.3494415, 81.4479065, -202.8344421, 204.2479248
3: -127.6544647, 68.1719131, -131.8819427, 70.3592148, -198.0136719, 200.0538177
4: -117.9137955, 91.0050430, -121.7929916, 93.9444122, -211.8582001, 212.7980347
5: -105.3133698, 82.1346741, -108.7460251, 84.8130188, -190.1263580, 190.8806763
6: -101.3108902, 98.0864639, -104.6058273, 101.2825317, -202.5934143, 202.6922607
7: -109.8791428, 92.7337799, -113.4803543, 95.7620163, -205.6411591, 206.2141418
8: -134.2003784, 93.1455078, -138.5416107, 96.0869446, -230.2873230, 231.6871185
9: -100.5049362, 99.8177338, -103.7757416, 103.0321503, -203.5370636, 203.5934753

Time for backsubstitution: 1.06 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=203.5233612060547
rel_dist={1: [-202.608202498089, 202.60820249808899]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1809.45 seconds
