## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 338.886556389
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-184.7211609, 146.8837128, -184.7211609, 146.8837128, -331.6048584, 331.6048584)
1: (-155.1049194, 130.5072479, -155.1049194, 130.5072479, -285.6120911, 285.6120911)
2: (-203.6430969, 131.9900055, -203.6430969, 131.9900055, -335.6330566, 335.6330566)
3: (-216.2613983, 114.2387161, -216.2613983, 114.2387161, -330.5000916, 330.5000916)
4: (-198.3916016, 151.6778870, -198.3916016, 151.6778870, -350.0694885, 350.0694885)
5: (-177.7157135, 138.1587830, -177.7157135, 138.1587830, -315.8745117, 315.8745117)
6: (-170.2996063, 163.9844208, -170.2996063, 163.9844208, -334.2840271, 334.2840271)
7: (-185.1827240, 156.0302124, -185.1827240, 156.0302124, -341.2129517, 341.2129517)
8: (-223.8251801, 153.0471344, -223.8251801, 153.0471344, -376.8723145, 376.8723145)
9: (-169.0843506, 166.1456757, -169.0843506, 166.1456757, -335.2300110, 335.2300110)

## BASE Result
execution time: IAR + LP analysis = 1.16 + 11.65 = 12.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -338.9345818, upper bound: 338.9345818


# Binary Search by BASE starts (time budget: 2687.19 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=341.21295166015625
rel_dist={7: [-338.93443485344073, 338.93443485344073]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=341.21295166015625
rel_dist={7: [-338.93400199054076, 338.93400199054076]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=341.21295166015625
rel_dist={7: [-338.93322554933513, 338.933225507835]}

## Binary Search Result
Binary search time: 44.91 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2642.28 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9266052, upper bound: 338.9262057
time: 8.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9249742, upper bound: 338.9249742
time: 8.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.38 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.38
Output dim: 7, lower bound: -338.9266052, upper bound: 338.9262057
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.38
Output dim: 7, lower bound: -338.9249742, upper bound: 338.9249742

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -175.0528107, 139.2155762, -184.7211609, 146.8837128, -321.9365234, 323.9367371
1: -147.0159912, 123.6377182, -155.1049194, 130.5072479, -277.5232239, 278.7425842
2: -192.9962158, 125.1462326, -203.6430969, 131.9900055, -324.9862061, 328.7893066
3: -204.8572540, 108.2642593, -216.2613983, 114.2387161, -319.0959778, 324.5256348
4: -187.9599152, 143.7206421, -198.3916016, 151.6778870, -339.6378174, 342.1122131
5: -168.3785706, 130.9101868, -177.7157135, 138.1587830, -306.5373535, 308.6258545
6: -161.3935852, 155.4163666, -170.2996063, 163.9844208, -325.3779907, 325.7159424
7: -175.4396667, 147.9055023, -185.1827240, 156.0302124, -331.4698792, 333.0882263
8: -212.2744598, 145.0615387, -223.8251801, 153.0471344, -365.3215942, 368.8866882
9: -160.2489624, 157.4055939, -169.0843506, 166.1456757, -326.3946533, 326.4899292

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9249742, upper bound: 338.9249742
time: 10.60 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9249742, upper bound: 338.9249742
time: 9.55 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -174.2116699, 138.5652008, -181.8791504, 144.6374512, -318.8490601, 320.4442444
1: -146.2442017, 122.9918976, -152.7155762, 128.4902344, -274.7344360, 275.7074585
2: -192.0296326, 124.5401611, -200.5099182, 129.9753113, -322.0049438, 325.0500488
3: -203.8439789, 107.7798386, -212.9136047, 112.4888916, -316.3328857, 320.6933899
4: -187.0102692, 142.9896851, -195.3255310, 149.3397522, -336.3500366, 338.3151855
5: -167.5493469, 130.2392883, -174.9758148, 136.0326233, -303.5819702, 305.2150879
6: -160.6019897, 154.6650085, -167.6797943, 161.4683075, -322.0703125, 322.3447876
7: -174.5427856, 147.1990662, -182.3173676, 153.6455994, -328.1883850, 329.5164185
8: -211.3038635, 144.3482513, -220.4187775, 150.7010498, -362.0048218, 364.7670288
9: -159.4755096, 156.5922089, -166.4912262, 163.5794220, -323.0549316, 323.0834045

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9151518, upper bound: 338.9147725
time: 10.44 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9168911, upper bound: 338.9168911
time: 8.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.61
Output dim: 7, lower bound: -338.9249742, upper bound: 338.9249742
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.61
Output dim: 7, lower bound: -338.9249742, upper bound: 338.9249742
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.61
Output dim: 7, lower bound: -338.9151518, upper bound: 338.9147725
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.61
Output dim: 7, lower bound: -338.9168911, upper bound: 338.9168911

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -175.0528107, 139.2155762, -175.0528107, 139.2155762, -314.2683716, 314.2683716
1: -147.0159912, 123.6377182, -147.0159912, 123.6377182, -270.6537170, 270.6537170
2: -192.9962158, 125.1462326, -192.9962158, 125.1462326, -318.1424561, 318.1424561
3: -204.8572540, 108.2642593, -204.8572540, 108.2642593, -313.1215210, 313.1215210
4: -187.9599152, 143.7206421, -187.9599152, 143.7206421, -331.6805420, 331.6805420
5: -168.3785706, 130.9101868, -168.3785706, 130.9101868, -299.2886353, 299.2886353
6: -161.3935852, 155.4163666, -161.3935852, 155.4163666, -316.8098755, 316.8099060
7: -175.4396667, 147.9055023, -175.4396667, 147.9055023, -323.3451538, 323.3451538
8: -212.2744598, 145.0615387, -212.2744598, 145.0615387, -357.3359375, 357.3359375
9: -160.2489624, 157.4055939, -160.2489624, 157.4055939, -317.6545410, 317.6545410

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9169651, upper bound: 338.9167605
time: 9.72 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9188925, upper bound: 338.9183506
time: 9.59 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -175.0528107, 139.2155762, -174.2116699, 138.5652008, -313.6180115, 313.4271545
1: -147.0159912, 123.6377182, -146.2442017, 122.9918976, -270.0078735, 269.8819275
2: -192.9962158, 125.1462326, -192.0296326, 124.5401611, -317.5363770, 317.1758728
3: -204.8572540, 108.2642593, -203.8439789, 107.7798386, -312.6370850, 312.1082458
4: -187.9599152, 143.7206421, -187.0102692, 142.9896851, -330.9495850, 330.7308960
5: -168.3785706, 130.9101868, -167.5493469, 130.2392883, -298.6177673, 298.4595337
6: -161.3935852, 155.4163666, -160.6019897, 154.6650085, -316.0585938, 316.0183411
7: -175.4396667, 147.9055023, -174.5427856, 147.1990662, -322.6387329, 322.4483032
8: -212.2744598, 145.0615387, -211.3038635, 144.3482513, -356.6227112, 356.3653564
9: -160.2489624, 157.4055939, -159.4755096, 156.5922089, -316.8411865, 316.8810730

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9169651, upper bound: 338.9167605
time: 8.83 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9188925, upper bound: 338.9183506
time: 9.92 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -173.8510437, 138.2803497, -165.7244873, 131.8782806, -305.7292786, 304.0048218
1: -145.9403687, 122.7355194, -139.1071930, 117.0119934, -262.9523621, 261.8426819
2: -191.6299286, 124.2804642, -182.6127472, 118.3476257, -309.9775391, 306.8931580
3: -203.4172363, 107.5582962, -193.8079987, 102.5716476, -305.9888916, 301.3663025
4: -186.6197205, 142.6915588, -177.8328247, 135.9905396, -322.6102295, 320.5243835
5: -167.2043610, 129.9703674, -159.5187531, 123.9909058, -291.1952515, 289.4891052
6: -160.2667847, 154.3461304, -152.6687622, 147.1881714, -307.4549561, 307.0148926
7: -174.1756744, 146.8945007, -165.8844147, 140.0066223, -314.1822510, 312.7788696
8: -210.8647766, 144.0520477, -200.7584686, 137.4377747, -348.3025513, 344.8104553
9: -159.1439514, 156.2646179, -151.6461334, 148.9108887, -308.0548401, 307.9107666

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9139625
time: 9.23 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9147725
time: 9.07 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -172.7469482, 137.4093781, -171.7309113, 136.6417389, -309.3886108, 309.1402893
1: -145.0157471, 121.9558411, -144.1984100, 121.3022690, -266.3180237, 266.1542358
2: -190.4129639, 123.4888840, -189.3087769, 122.6860504, -313.0989990, 312.7976074
3: -202.1149597, 106.8814087, -200.9283447, 106.2701721, -308.3851318, 307.8096924
4: -185.4255066, 141.7812653, -184.3309174, 140.9515839, -326.3770752, 326.1121216
5: -166.1475220, 129.1485291, -165.2789307, 128.4877472, -294.6352539, 294.4274597
6: -159.2432404, 153.3723602, -158.2507935, 152.5194397, -311.7626953, 311.6231079
7: -173.0608063, 145.9641724, -172.0395813, 145.0918121, -318.1526184, 318.0037537
8: -209.5289917, 143.1461029, -208.1081390, 142.3562164, -351.8851624, 351.2542419
9: -158.1328888, 155.2665100, -157.1739807, 154.3698578, -312.5027466, 312.4404907

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9082753, upper bound: 338.9073624
time: 8.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9049017, upper bound: 338.9049017
time: 8.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.61 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.61
Output dim: 7, lower bound: -338.9169651, upper bound: 338.9167605
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.61
Output dim: 7, lower bound: -338.9188925, upper bound: 338.9183506
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.61
Output dim: 7, lower bound: -338.9169651, upper bound: 338.9167605
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.61
Output dim: 7, lower bound: -338.9188925, upper bound: 338.9183506
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.61
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9139625
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.61
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9147725
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.61
Output dim: 7, lower bound: -338.9082753, upper bound: 338.9073624
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.61
Output dim: 7, lower bound: -338.9049017, upper bound: 338.9049017

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -159.0477905, 126.5746765, -174.6963348, 138.9339905, -297.9817810, 301.2709961
1: -133.5379333, 112.2659988, -146.7157135, 123.3843689, -256.9222717, 258.9817200
2: -175.2654419, 113.6256180, -192.6011810, 124.8894882, -300.1549377, 306.2267761
3: -185.9284668, 98.4391327, -204.4355011, 108.0453110, -293.9737854, 302.8746338
4: -170.6317749, 130.4969482, -187.5739441, 143.4260559, -314.0578003, 318.0708923
5: -153.0649414, 118.9783096, -168.0374603, 130.6443634, -283.7092896, 287.0157776
6: -146.5221100, 141.2702484, -161.0622101, 155.1012573, -301.6233521, 302.3324280
7: -159.1599731, 134.3921051, -175.0769043, 147.6044617, -306.7644348, 309.4689941
8: -192.7991486, 131.9246674, -211.8404236, 144.7688141, -337.5679321, 343.7650452
9: -145.5430756, 142.8740082, -159.9213715, 157.0818939, -302.6249695, 302.7953186

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9203017, upper bound: 338.9203017
time: 8.55 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9203017, upper bound: 338.9212120
time: 9.48 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -164.8116150, 131.1473083, -173.5612793, 138.0385284, -302.8500977, 304.7085876
1: -138.4195709, 116.3805389, -145.7643433, 122.5820389, -261.0016174, 262.1448669
2: -181.6929474, 117.7904129, -191.3496399, 124.0761566, -305.7691040, 309.1400452
3: -192.7609100, 101.9880295, -203.0961609, 107.3491135, -300.1100159, 305.0841980
4: -176.8644409, 135.2548065, -186.3459015, 142.4898682, -319.3543091, 321.6007080
5: -158.5954437, 123.2924500, -166.9512939, 129.7994080, -288.3948059, 290.2437439
6: -151.8781281, 146.3838959, -160.0100403, 154.0997162, -305.9777832, 306.3938904
7: -165.0650177, 139.2704468, -173.9297485, 146.6480713, -311.7130737, 313.2001343
8: -199.8519135, 136.6371613, -210.4671173, 143.8373413, -343.6892395, 347.1042786
9: -150.8415680, 148.1102753, -158.8811188, 156.0553436, -306.8969116, 306.9913940

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9212120, upper bound: 338.9210058
time: 9.32 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9212120, upper bound: 338.9210058
time: 9.08 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -159.0477905, 126.5746765, -173.8510437, 138.2803497, -297.3281250, 300.4256897
1: -133.5379333, 112.2659988, -145.9403687, 122.7355194, -256.2734070, 258.2063599
2: -175.2654419, 113.6256180, -191.6299286, 124.2804642, -299.5458984, 305.2555237
3: -185.9284668, 98.4391327, -203.4172363, 107.5582962, -293.4867249, 301.8563843
4: -170.6317749, 130.4969482, -186.6197205, 142.6915588, -313.3233032, 317.1166382
5: -153.0649414, 118.9783096, -167.2043610, 129.9703674, -283.0353088, 286.1826782
6: -146.5221100, 141.2702484, -160.2667847, 154.3461304, -300.8682251, 301.5370483
7: -159.1599731, 134.3921051, -174.1756744, 146.8945007, -306.0544739, 308.5677490
8: -192.7991486, 131.9246674, -210.8647766, 144.0520477, -336.8511658, 342.7893982
9: -145.5430756, 142.8740082, -159.1439514, 156.2646179, -301.8076782, 302.0178833

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9159776, upper bound: 338.9155320
time: 9.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9159776, upper bound: 338.9167605
time: 9.22 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -164.8116150, 131.1473083, -172.7469482, 137.4093781, -302.2210083, 303.8941956
1: -138.4195709, 116.3805389, -145.0157471, 121.9558411, -260.3754272, 261.3963013
2: -181.6929474, 117.7904129, -190.4129639, 123.4888840, -305.1818237, 308.2033691
3: -192.7609100, 101.9880295, -202.1149597, 106.8814087, -299.6422729, 304.1029968
4: -176.8644409, 135.2548065, -185.4255066, 141.7812653, -318.6456604, 320.6802673
5: -158.5954437, 123.2924500, -166.1475220, 129.1485291, -287.7439575, 289.4399719
6: -151.8781281, 146.3838959, -159.2432404, 153.3723602, -305.2504883, 305.6271057
7: -165.0650177, 139.2704468, -173.0608063, 145.9641724, -311.0291443, 312.3312378
8: -199.8519135, 136.6371613, -209.5289917, 143.1461029, -342.9980164, 346.1661072
9: -150.8415680, 148.1102753, -158.1328888, 155.2665100, -306.1080933, 306.2431641

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9089926, upper bound: 338.9095304
time: 9.98 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9064871, upper bound: 338.9060734
time: 11.11 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -157.9628143, 125.7313690, -165.7244873, 131.8782806, -289.8410950, 291.4558716
1: -132.5550995, 111.4434662, -139.1071930, 117.0119934, -249.5670929, 250.5506592
2: -174.0249023, 112.8431702, -182.6127472, 118.3476257, -292.3724976, 295.4559021
3: -184.6233063, 97.8029480, -193.8079987, 102.5716476, -287.1949463, 291.6109619
4: -169.4144592, 129.5594025, -177.8328247, 135.9905396, -305.4049377, 307.3921814
5: -152.0036469, 118.1241226, -159.5187531, 123.9909058, -275.9945374, 277.6428833
6: -145.5032806, 140.3009796, -152.6687622, 147.1881714, -292.6914368, 292.9697266
7: -158.0111694, 133.4749603, -165.8844147, 140.0066223, -298.0177917, 299.3593445
8: -191.5286407, 131.0045776, -200.7584686, 137.4377747, -328.9664307, 331.7630005
9: -144.5401917, 141.8323669, -151.6461334, 148.9108887, -293.4510498, 293.4784851

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9139625
time: 9.30 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9139625
time: 9.15 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -164.1298828, 130.6184845, -165.7244873, 131.8782806, -296.0081787, 296.3429565
1: -137.7825012, 115.8496323, -139.1071930, 117.0119934, -254.7944946, 254.9568176
2: -180.8995819, 117.2933426, -182.6127472, 118.3476257, -299.2471924, 299.9060669
3: -191.9367828, 101.5976028, -193.8079987, 102.5716476, -294.5084229, 295.4056091
4: -176.0870514, 134.6571503, -177.8328247, 135.9905396, -312.0775146, 312.4899902
5: -157.9129639, 122.7383423, -159.5187531, 123.9909058, -281.9038696, 282.2570801
6: -151.2293854, 145.7716064, -152.6687622, 147.1881714, -298.4175110, 298.4403687
7: -164.3294678, 138.6969299, -165.8844147, 140.0066223, -304.3360596, 304.5812988
8: -199.0671692, 136.0540009, -200.7584686, 137.4377747, -336.5049133, 336.8124390
9: -150.2177582, 147.4386444, -151.6461334, 148.9108887, -299.1286621, 299.0847778

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9147725
time: 8.38 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9147725
time: 8.47 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -169.8241882, 135.0921021, -171.7309113, 136.6417389, -306.4659119, 306.8229980
1: -142.5433502, 119.8856583, -144.1984100, 121.3022690, -263.8455200, 264.0840759
2: -187.1743774, 121.4106140, -189.3087769, 122.6860504, -309.8604126, 310.7193298
3: -198.6752777, 105.0696945, -200.9283447, 106.2701721, -304.9454346, 305.9980469
4: -182.2491760, 139.3564301, -184.3309174, 140.9515839, -323.2007446, 323.6873474
5: -163.3069305, 126.9435196, -165.2789307, 128.4877472, -291.7946777, 292.2224426
6: -156.5330200, 150.7680817, -158.2507935, 152.5194397, -309.0524597, 309.0187988
7: -170.0943298, 143.4778442, -172.0395813, 145.0918121, -315.1861572, 315.5174255
8: -206.0367126, 140.7439270, -208.1081390, 142.3562164, -348.3929443, 348.8520508
9: -155.4245605, 152.6086884, -157.1739807, 154.3698578, -309.7943726, 309.7826538

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9082753, upper bound: 338.9073624
time: 11.41 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9082753, upper bound: 338.9073624
time: 9.27 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -174.7698669, 139.0100250, -170.6146545, 135.7583313, -310.5281677, 309.6246948
1: -146.6282654, 123.3069153, -143.2525177, 120.5115433, -267.1397400, 266.5594177
2: -192.5989532, 124.9205399, -188.0730743, 121.8904648, -314.4893799, 312.9936218
3: -204.4269104, 108.0896072, -199.6124115, 105.5799103, -310.0068359, 307.7020264
4: -187.5210724, 143.3385925, -183.1195374, 140.0269928, -327.5480652, 326.4580994
5: -167.9794769, 130.5569916, -164.1949768, 127.6460800, -295.6255493, 294.7519531
6: -161.0521545, 155.1096802, -157.2160950, 151.5269165, -312.5789795, 312.3257751
7: -175.0100098, 147.5576019, -170.9067078, 144.1434174, -319.1534424, 318.4642029
8: -212.0679169, 144.7861786, -206.7707214, 141.4435272, -353.5114441, 351.5568848
9: -159.8668823, 156.9207153, -156.1426086, 153.3563080, -313.2232056, 313.0632629

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9049016, upper bound: 338.9049017
time: 8.96 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9049017, upper bound: 338.9049017
time: 10.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.67 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9203017, upper bound: 338.9203017
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9203017, upper bound: 338.9212120
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9212120, upper bound: 338.9210058
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9212120, upper bound: 338.9210058
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9159776, upper bound: 338.9155320
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9159776, upper bound: 338.9167605
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9089926, upper bound: 338.9095304
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9064871, upper bound: 338.9060734
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9139625
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9139625
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9147725
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9139625, upper bound: 338.9147725
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9082753, upper bound: 338.9073624
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9082753, upper bound: 338.9073624
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9049016, upper bound: 338.9049017
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 7, lower bound: -338.9049017, upper bound: 338.9049017

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -159.0477905, 126.5746765, -159.0477905, 126.5746765, -285.6224670, 285.6224670
1: -133.5379333, 112.2659988, -133.5379333, 112.2659988, -245.8039246, 245.8039246
2: -175.2654419, 113.6256180, -175.2654419, 113.6256180, -288.8910522, 288.8910522
3: -185.9284668, 98.4391327, -185.9284668, 98.4391327, -284.3675842, 284.3675842
4: -170.6317749, 130.4969482, -170.6317749, 130.4969482, -301.1286316, 301.1286316
5: -153.0649414, 118.9783096, -153.0649414, 118.9783096, -272.0432434, 272.0432434
6: -146.5221100, 141.2702484, -146.5221100, 141.2702484, -287.7923584, 287.7923584
7: -159.1599731, 134.3921051, -159.1599731, 134.3921051, -293.5520630, 293.5520630
8: -192.7991486, 131.9246674, -192.7991486, 131.9246674, -324.7237854, 324.7237854
9: -145.5430756, 142.8740082, -145.5430756, 142.8740082, -288.4170532, 288.4170532

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9114661, upper bound: 338.9105089
time: 10.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9081566, upper bound: 338.9080833
time: 9.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -159.0477905, 126.5746765, -164.8116150, 131.1473083, -290.1950989, 291.3862915
1: -133.5379333, 112.2659988, -138.4195709, 116.3805389, -249.9184723, 250.6855774
2: -175.2654419, 113.6256180, -181.6929474, 117.7904129, -293.0558472, 295.3185730
3: -185.9284668, 98.4391327, -192.7609100, 101.9880295, -287.9164734, 291.2000427
4: -170.6317749, 130.4969482, -176.8644409, 135.2548065, -305.8865051, 307.3613892
5: -153.0649414, 118.9783096, -158.5954437, 123.2924500, -276.3573914, 277.5737610
6: -146.5221100, 141.2702484, -151.8781281, 146.3838959, -292.9060059, 293.1483765
7: -159.1599731, 134.3921051, -165.0650177, 139.2704468, -298.4304199, 299.4571228
8: -192.7991486, 131.9246674, -199.8519135, 136.6371613, -329.4363098, 331.7765503
9: -145.5430756, 142.8740082, -150.8415680, 148.1102753, -293.6533508, 293.7154846

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9114661, upper bound: 338.9110089
time: 9.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9081566, upper bound: 338.9086118
time: 8.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -164.8116150, 131.1473083, -159.0477905, 126.5746765, -291.3862915, 290.1950989
1: -138.4195709, 116.3805389, -133.5379333, 112.2659988, -250.6855774, 249.9184723
2: -181.6929474, 117.7904129, -175.2654419, 113.6256180, -295.3185730, 293.0558472
3: -192.7609100, 101.9880295, -185.9284668, 98.4391327, -291.2000427, 287.9164734
4: -176.8644409, 135.2548065, -170.6317749, 130.4969482, -307.3613892, 305.8865051
5: -158.5954437, 123.2924500, -153.0649414, 118.9783096, -277.5737610, 276.3573914
6: -151.8781281, 146.3838959, -146.5221100, 141.2702484, -293.1483765, 292.9060059
7: -165.0650177, 139.2704468, -159.1599731, 134.3921051, -299.4571228, 298.4304199
8: -199.8519135, 136.6371613, -192.7991486, 131.9246674, -331.7765503, 329.4363098
9: -150.8415680, 148.1102753, -145.5430756, 142.8740082, -293.7154846, 293.6533508

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9123706, upper bound: 338.9112212
time: 9.30 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9086118, upper bound: 338.9086016
time: 9.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -164.8116150, 131.1473083, -164.8116150, 131.1473083, -295.9589233, 295.9589233
1: -138.4195709, 116.3805389, -138.4195709, 116.3805389, -254.8001099, 254.8001099
2: -181.6929474, 117.7904129, -181.6929474, 117.7904129, -299.4833679, 299.4833679
3: -192.7609100, 101.9880295, -192.7609100, 101.9880295, -294.7489014, 294.7489014
4: -176.8644409, 135.2548065, -176.8644409, 135.2548065, -312.1192627, 312.1192627
5: -158.5954437, 123.2924500, -158.5954437, 123.2924500, -281.8878784, 281.8878784
6: -151.8781281, 146.3838959, -151.8781281, 146.3838959, -298.2619934, 298.2619934
7: -165.0650177, 139.2704468, -165.0650177, 139.2704468, -304.3354187, 304.3354187
8: -199.8519135, 136.6371613, -199.8519135, 136.6371613, -336.4890747, 336.4890747
9: -150.8415680, 148.1102753, -150.8415680, 148.1102753, -298.9518433, 298.9518433

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9123706, upper bound: 338.9124252
time: 9.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9086118, upper bound: 338.9097624
time: 8.31 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -159.0477905, 126.5746765, -157.9628143, 125.7313690, -284.7791748, 284.5374756
1: -133.5379333, 112.2659988, -132.5550995, 111.4434662, -244.9813995, 244.8210907
2: -175.2654419, 113.6256180, -174.0249023, 112.8431702, -288.1086121, 287.6504822
3: -185.9284668, 98.4391327, -184.6233063, 97.8029480, -283.7314148, 283.0624390
4: -170.6317749, 130.4969482, -169.4144592, 129.5594025, -300.1911011, 299.9113464
5: -153.0649414, 118.9783096, -152.0036469, 118.1241226, -271.1890564, 270.9819641
6: -146.5221100, 141.2702484, -145.5032806, 140.3009796, -286.8230896, 286.7735291
7: -159.1599731, 134.3921051, -158.0111694, 133.4749603, -292.6349487, 292.4032593
8: -192.7991486, 131.9246674, -191.5286407, 131.0045776, -323.8037109, 323.4532776
9: -145.5430756, 142.8740082, -144.5401917, 141.8323669, -287.3754272, 287.4140320

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9077591, upper bound: 338.9064776
time: 10.21 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9045815, upper bound: 338.9041925
time: 10.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -159.0477905, 126.5746765, -164.1298828, 130.6184845, -289.6662598, 290.7045593
1: -133.5379333, 112.2659988, -137.7825012, 115.8496323, -249.3875732, 250.0484924
2: -175.2654419, 113.6256180, -180.8995819, 117.2933426, -292.5587769, 294.5251770
3: -185.9284668, 98.4391327, -191.9367828, 101.5976028, -287.5260315, 290.3759155
4: -170.6317749, 130.4969482, -176.0870514, 134.6571503, -305.2888794, 306.5839233
5: -153.0649414, 118.9783096, -157.9129639, 122.7383423, -275.8032837, 276.8912659
6: -146.5221100, 141.2702484, -151.2293854, 145.7716064, -292.2937012, 292.4996338
7: -159.1599731, 134.3921051, -164.3294678, 138.6969299, -297.8569031, 298.7215576
8: -192.7991486, 131.9246674, -199.0671692, 136.0540009, -328.8531494, 330.9917603
9: -145.5430756, 142.8740082, -150.2177582, 147.4386444, -292.9817200, 293.0917053

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9077591, upper bound: 338.9071814
time: 9.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9045815, upper bound: 338.9049985
time: 9.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -164.8116150, 131.1473083, -169.8241882, 135.0921021, -299.9037170, 300.9714966
1: -138.4195709, 116.3805389, -142.5433502, 119.8856583, -258.3052063, 258.9238586
2: -181.6929474, 117.7904129, -187.1743774, 121.4106140, -303.1035461, 304.9647827
3: -192.7609100, 101.9880295, -198.6752777, 105.0696945, -297.8305664, 300.6632996
4: -176.8644409, 135.2548065, -182.2491760, 139.3564301, -316.2208862, 317.5039673
5: -158.5954437, 123.2924500, -163.3069305, 126.9435196, -285.5389709, 286.5993652
6: -151.8781281, 146.3838959, -156.5330200, 150.7680817, -302.6462097, 302.9169006
7: -165.0650177, 139.2704468, -170.0943298, 143.4778442, -308.5428467, 309.3647461
8: -199.8519135, 136.6371613, -206.0367126, 140.7439270, -340.5958252, 342.6738892
9: -150.8415680, 148.1102753, -155.4245605, 152.6086884, -303.4502258, 303.5348511

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9064871, upper bound: 338.9060734
time: 8.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9064871, upper bound: 338.9060734
time: 8.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -163.6865692, 130.2568817, -174.7698669, 139.0100250, -302.6965942, 305.0267029
1: -137.4659271, 115.5834274, -146.6282654, 123.3069153, -260.7727966, 262.2116394
2: -180.4478912, 116.9884338, -192.5989532, 124.9205399, -305.3683777, 309.5874023
3: -191.4344635, 101.2922440, -204.4269104, 108.0896072, -299.5240784, 305.7191467
4: -175.6437531, 134.3225861, -187.5210724, 143.3385925, -318.9823303, 321.8436279
5: -157.5027924, 122.4437943, -167.9794769, 130.5569916, -288.0597839, 290.4232788
6: -150.8353729, 145.3833160, -161.0521545, 155.1096802, -305.9450073, 306.4354248
7: -163.9235077, 138.3147583, -175.0100098, 147.5576019, -311.4810791, 313.3247681
8: -198.5040741, 135.7172852, -212.0679169, 144.7861786, -343.2902527, 347.7852173
9: -149.8022461, 147.0886078, -159.8668823, 156.9207153, -306.7229614, 306.9554749

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9064871, upper bound: 338.9060734
time: 9.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9064871, upper bound: 338.9060734
time: 9.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -157.9628143, 125.7313690, -159.0477905, 126.5746765, -284.5374756, 284.7791748
1: -132.5550995, 111.4434662, -133.5379333, 112.2659988, -244.8210907, 244.9813995
2: -174.0249023, 112.8431702, -175.2654419, 113.6256180, -287.6504822, 288.1086121
3: -184.6233063, 97.8029480, -185.9284668, 98.4391327, -283.0624390, 283.7314148
4: -169.4144592, 129.5594025, -170.6317749, 130.4969482, -299.9113464, 300.1911011
5: -152.0036469, 118.1241226, -153.0649414, 118.9783096, -270.9819641, 271.1890564
6: -145.5032806, 140.3009796, -146.5221100, 141.2702484, -286.7735291, 286.8230896
7: -158.0111694, 133.4749603, -159.1599731, 134.3921051, -292.4032593, 292.6349487
8: -191.5286407, 131.0045776, -192.7991486, 131.9246674, -323.4532776, 323.8037109
9: -144.5401917, 141.8323669, -145.5430756, 142.8740082, -287.4140320, 287.3754272

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9058654, upper bound: 338.9050215
time: 9.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9028502, upper bound: 338.9028502
time: 10.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -157.9628143, 125.7313690, -157.9628143, 125.7313690, -283.6941833, 283.6941833
1: -132.5550995, 111.4434662, -132.5550995, 111.4434662, -243.9985504, 243.9985504
2: -174.0249023, 112.8431702, -174.0249023, 112.8431702, -286.8680725, 286.8680725
3: -184.6233063, 97.8029480, -184.6233063, 97.8029480, -282.4262390, 282.4262390
4: -169.4144592, 129.5594025, -169.4144592, 129.5594025, -298.9738159, 298.9738159
5: -152.0036469, 118.1241226, -152.0036469, 118.1241226, -270.1277771, 270.1277771
6: -145.5032806, 140.3009796, -145.5032806, 140.3009796, -285.8042603, 285.8042603
7: -158.0111694, 133.4749603, -158.0111694, 133.4749603, -291.4860840, 291.4860840
8: -191.5286407, 131.0045776, -191.5286407, 131.0045776, -322.5332031, 322.5332031
9: -144.5401917, 141.8323669, -144.5401917, 141.8323669, -286.3724670, 286.3724670

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9058654, upper bound: 338.9050215
time: 10.24 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9028502, upper bound: 338.9028502
time: 9.07 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -164.1298828, 130.6184845, -159.0477905, 126.5746765, -290.7045593, 289.6662598
1: -137.7825012, 115.8496323, -133.5379333, 112.2659988, -250.0484924, 249.3875732
2: -180.8995819, 117.2933426, -175.2654419, 113.6256180, -294.5251770, 292.5587769
3: -191.9367828, 101.5976028, -185.9284668, 98.4391327, -290.3759155, 287.5260315
4: -176.0870514, 134.6571503, -170.6317749, 130.4969482, -306.5839233, 305.2888794
5: -157.9129639, 122.7383423, -153.0649414, 118.9783096, -276.8912659, 275.8032837
6: -151.2293854, 145.7716064, -146.5221100, 141.2702484, -292.4996338, 292.2937012
7: -164.3294678, 138.6969299, -159.1599731, 134.3921051, -298.7215576, 297.8569031
8: -199.0671692, 136.0540009, -192.7991486, 131.9246674, -330.9917603, 328.8531494
9: -150.2177582, 147.4386444, -145.5430756, 142.8740082, -293.0917053, 292.9817200

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9071345, upper bound: 338.9059487
time: 10.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9035771, upper bound: 338.9034927
time: 9.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -164.1298828, 130.6184845, -157.9628143, 125.7313690, -289.8612671, 288.5812988
1: -137.7825012, 115.8496323, -132.5550995, 111.4434662, -249.2259674, 248.4047241
2: -180.8995819, 117.2933426, -174.0249023, 112.8431702, -293.7427368, 291.3182068
3: -191.9367828, 101.5976028, -184.6233063, 97.8029480, -289.7397461, 286.2208557
4: -176.0870514, 134.6571503, -169.4144592, 129.5594025, -305.6463623, 304.0715942
5: -157.9129639, 122.7383423, -152.0036469, 118.1241226, -276.0370789, 274.7420044
6: -151.2293854, 145.7716064, -145.5032806, 140.3009796, -291.5303650, 291.2748413
7: -164.3294678, 138.6969299, -158.0111694, 133.4749603, -297.8043823, 296.7080688
8: -199.0671692, 136.0540009, -191.5286407, 131.0045776, -330.0717163, 327.5826416
9: -150.2177582, 147.4386444, -144.5401917, 141.8323669, -292.0501099, 291.9788208

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9071345, upper bound: 338.9059487
time: 12.02 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9035771, upper bound: 338.9034927
time: 9.35 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -169.8241882, 135.0921021, -164.8116150, 131.1473083, -300.9714966, 299.9037170
1: -142.5433502, 119.8856583, -138.4195709, 116.3805389, -258.9238586, 258.3052063
2: -187.1743774, 121.4106140, -181.6929474, 117.7904129, -304.9647827, 303.1035461
3: -198.6752777, 105.0696945, -192.7609100, 101.9880295, -300.6632996, 297.8305664
4: -182.2491760, 139.3564301, -176.8644409, 135.2548065, -317.5039673, 316.2208862
5: -163.3069305, 126.9435196, -158.5954437, 123.2924500, -286.5993652, 285.5389709
6: -156.5330200, 150.7680817, -151.8781281, 146.3838959, -302.9169006, 302.6462097
7: -170.0943298, 143.4778442, -165.0650177, 139.2704468, -309.3647461, 308.5428467
8: -206.0367126, 140.7439270, -199.8519135, 136.6371613, -342.6738892, 340.5958252
9: -155.4245605, 152.6086884, -150.8415680, 148.1102753, -303.5348511, 303.4502258

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9058568, upper bound: 338.9056780
time: 9.49 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9058568, upper bound: 338.9073624
time: 9.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -169.8241882, 135.0921021, -164.1298828, 130.6184845, -300.4426880, 299.2219849
1: -142.5433502, 119.8856583, -137.7825012, 115.8496323, -258.3929443, 257.6681519
2: -187.1743774, 121.4106140, -180.8995819, 117.2933426, -304.4677124, 302.3101807
3: -198.6752777, 105.0696945, -191.9367828, 101.5976028, -300.2728882, 297.0064697
4: -182.2491760, 139.3564301, -176.0870514, 134.6571503, -316.9063110, 315.4434509
5: -163.3069305, 126.9435196, -157.9129639, 122.7383423, -286.0452881, 284.8564758
6: -156.5330200, 150.7680817, -151.2293854, 145.7716064, -302.3045959, 301.9974365
7: -170.0943298, 143.4778442, -164.3294678, 138.6969299, -308.7912292, 307.8073120
8: -206.0367126, 140.7439270, -199.0671692, 136.0540009, -342.0906982, 339.8110962
9: -155.4245605, 152.6086884, -150.2177582, 147.4386444, -302.8632202, 302.8264465

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9058568, upper bound: 338.9056780
time: 9.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9058568, upper bound: 338.9073624
time: 9.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.21 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9114661, upper bound: 338.9105089
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9081566, upper bound: 338.9080833
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9114661, upper bound: 338.9110089
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9081566, upper bound: 338.9086118
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9123706, upper bound: 338.9112212
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9086118, upper bound: 338.9086016
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9123706, upper bound: 338.9124252
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9086118, upper bound: 338.9097624
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9077591, upper bound: 338.9064776
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9045815, upper bound: 338.9041925
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9077591, upper bound: 338.9071814
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9045815, upper bound: 338.9049985
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9064871, upper bound: 338.9060734
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9064871, upper bound: 338.9060734
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9064871, upper bound: 338.9060734
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9064871, upper bound: 338.9060734
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9058654, upper bound: 338.9050215
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9028502, upper bound: 338.9028502
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9058654, upper bound: 338.9050215
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9028502, upper bound: 338.9028502
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9071345, upper bound: 338.9059487
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9035771, upper bound: 338.9034927
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9071345, upper bound: 338.9059487
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9035771, upper bound: 338.9034927
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9058568, upper bound: 338.9056780
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9058568, upper bound: 338.9073624
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9058568, upper bound: 338.9056780
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 7, lower bound: -338.9058568, upper bound: 338.9073624
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.21
Output dim: 7, lower bound: -338.9049016, upper bound: 338.9049017
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.21
Output dim: 7, lower bound: -338.9049017, upper bound: 338.9049017
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=341.21295166015625
rel_dist={7: [-338.93443485344073, 338.93443485344073]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9250151, upper bound: 338.9247111
time: 10.14 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9240509, upper bound: 338.9240509
time: 9.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.08 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.08
Output dim: 7, lower bound: -338.9250151, upper bound: 338.9247111
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.08
Output dim: 7, lower bound: -338.9240509, upper bound: 338.9240509

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -175.0528107, 139.2155762, -182.4907990, 145.1144562, -320.1672668, 321.7062988
1: -147.0159912, 123.6377182, -153.2394409, 128.9224091, -275.9383545, 276.8771362
2: -192.9962158, 125.1462326, -201.1867523, 130.4115753, -323.4077759, 326.3329773
3: -204.8572540, 108.2642593, -213.6303864, 112.8603973, -317.7176514, 321.8946228
4: -187.9599152, 143.7206421, -195.9849701, 149.8422699, -337.8021851, 339.7055664
5: -168.3785706, 130.9101868, -175.5613251, 136.4862823, -304.8647766, 306.4714355
6: -161.3935852, 155.4163666, -168.2453308, 162.0076294, -323.4011841, 323.6616211
7: -175.4396667, 147.9055023, -182.9352264, 154.1561584, -329.5957947, 330.8407288
8: -212.2744598, 145.0615387, -221.1602631, 151.2046814, -363.4790344, 366.2217407
9: -160.2489624, 157.4055939, -167.0460510, 164.1296234, -324.3786011, 324.4515991

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9145251, upper bound: 338.9142257
time: 10.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9169111, upper bound: 338.9166381
time: 10.41 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -174.2116699, 138.5652008, -177.5824890, 141.2422485, -315.4538574, 316.1476440
1: -146.2442017, 122.9918976, -149.1007690, 125.4422684, -271.6864624, 272.0926514
2: -192.0296326, 124.5401611, -195.7722473, 126.9285583, -318.9581909, 320.3123169
3: -203.8439789, 107.7798386, -207.8542175, 109.8452454, -313.6892090, 315.6340027
4: -187.0102692, 142.9896851, -190.6895294, 145.8025665, -332.8128357, 333.6791992
5: -167.5493469, 130.2392883, -170.8345337, 132.8193512, -300.3687134, 301.0737610
6: -160.6019897, 154.6650085, -163.7191620, 157.6641235, -318.2661133, 318.3841553
7: -174.5427856, 147.1990662, -177.9852448, 150.0412445, -324.5840454, 325.1842957
8: -211.3038635, 144.3482513, -215.2672424, 147.1535339, -358.4573669, 359.6154785
9: -159.4755096, 156.5922089, -162.5710907, 159.7001648, -319.1756287, 319.1632996

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9136560, upper bound: 338.9134915
time: 13.05 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9157789, upper bound: 338.9157789
time: 9.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.59 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.59
Output dim: 7, lower bound: -338.9145251, upper bound: 338.9142257
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.59
Output dim: 7, lower bound: -338.9169111, upper bound: 338.9166381
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.59
Output dim: 7, lower bound: -338.9136560, upper bound: 338.9134915
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.59
Output dim: 7, lower bound: -338.9157789, upper bound: 338.9157789

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -168.8791656, 134.3389587, -166.3823395, 132.3914948, -301.2706299, 300.7212524
1: -141.8162689, 119.2501144, -139.6709900, 117.4768524, -259.2931213, 258.9211121
2: -186.1546936, 120.7008820, -183.3416138, 118.8172531, -304.9719543, 304.0424500
3: -197.5544739, 104.4732666, -194.5800323, 102.9707870, -300.5252686, 299.0532837
4: -181.2746124, 138.6190491, -178.5437012, 136.5323639, -317.8069763, 317.1627502
5: -162.4708099, 126.3067474, -160.1475983, 124.4782486, -286.9490662, 286.4543457
6: -155.6558990, 149.9589691, -153.2779999, 147.7692566, -303.4251099, 303.2369690
7: -169.1580505, 142.6916504, -166.5496826, 140.5548553, -309.7128906, 309.2413330
8: -204.7592163, 139.9923553, -201.5581818, 137.9819794, -342.7412109, 341.5505371
9: -154.5753021, 151.7998657, -152.2440033, 149.5039062, -304.0791931, 304.0438538

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9053218, upper bound: 338.9047511
time: 10.51 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9035341, upper bound: 338.9034488
time: 10.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -169.6726074, 134.9701996, -172.2980042, 137.0852814, -306.7578735, 307.2681885
1: -142.5026398, 119.8315735, -144.6846771, 121.7032166, -264.2058105, 264.5162354
2: -187.0585785, 121.2878113, -189.9389496, 123.0917358, -310.1503296, 311.2267151
3: -198.5079193, 104.9637070, -201.5921631, 106.6143265, -305.1222534, 306.5558472
4: -182.1391296, 139.2817993, -184.9445190, 141.4182739, -323.5574036, 324.2263184
5: -163.2306061, 126.9033966, -165.8235016, 128.9080505, -292.1386719, 292.7268982
6: -156.4049225, 150.6681824, -158.7758179, 153.0204010, -309.4253235, 309.4440002
7: -169.9953156, 143.3706512, -172.6132965, 145.5645905, -315.5598755, 315.9839172
8: -205.7588806, 140.6452026, -208.7986450, 142.8225250, -348.5814209, 349.4438171
9: -155.3152618, 152.5360718, -157.6873322, 154.8800659, -310.1953125, 310.2233582

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9067474, upper bound: 338.9062282
time: 10.72 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9050195, upper bound: 338.9048543
time: 10.22 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -167.9573975, 133.6249390, -161.3927612, 128.4554901, -296.4128113, 295.0177002
1: -140.9746399, 118.5463181, -135.4620819, 113.9377747, -254.9124146, 254.0083923
2: -185.0982666, 120.0370483, -177.8349609, 115.2764435, -300.3746948, 297.8719788
3: -196.4444275, 103.9381561, -188.7057343, 99.9062881, -296.3506775, 292.6438599
4: -180.2368774, 137.8202057, -173.1585693, 132.4234619, -312.6602783, 310.9787598
5: -161.5663910, 125.5750580, -155.3448334, 120.7511902, -282.3175659, 280.9198914
6: -154.7899017, 149.1352539, -148.6736145, 143.3529816, -298.1428833, 297.8087769
7: -168.1774445, 141.9167786, -161.5151520, 136.3732147, -304.5506592, 303.4318848
8: -203.6903229, 139.2111816, -195.5634003, 133.8597717, -337.5501099, 334.7745972
9: -153.7259064, 150.9107666, -147.6943817, 144.9985046, -298.7243652, 298.6051025

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9045511, upper bound: 338.9040846
time: 10.84 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9027031
time: 10.04 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -168.9223938, 134.3914185, -167.4692078, 133.2733002, -302.1956787, 301.8606262
1: -141.8089142, 119.2519836, -140.6113281, 118.2780838, -260.0870056, 259.8633118
2: -186.1928406, 120.7455368, -184.6084290, 119.6623993, -305.8551941, 305.3539429
3: -197.6020813, 104.5349274, -195.9100800, 103.6467438, -301.2488403, 300.4450073
4: -181.2879486, 138.6260529, -179.7308655, 137.4436188, -318.7315674, 318.3569336
5: -162.4869080, 126.3003845, -161.1688538, 125.2989349, -287.7858276, 287.4692383
6: -155.6964111, 149.9972839, -154.3202972, 148.7449036, -304.4412231, 304.3175659
7: -169.1927795, 142.7407227, -167.7406921, 141.5145721, -310.7073059, 310.4814148
8: -204.8967285, 140.0064392, -202.9951630, 138.8392639, -343.7359924, 343.0015869
9: -154.6267395, 151.8057556, -153.2861786, 150.5200500, -305.1467896, 305.0919189

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9059511, upper bound: 338.9055901
time: 10.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329
time: 9.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.17 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.17
Output dim: 7, lower bound: -338.9053218, upper bound: 338.9047511
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.17
Output dim: 7, lower bound: -338.9035341, upper bound: 338.9034488
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.17
Output dim: 7, lower bound: -338.9067474, upper bound: 338.9062282
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.17
Output dim: 7, lower bound: -338.9050195, upper bound: 338.9048543
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.17
Output dim: 7, lower bound: -338.9045511, upper bound: 338.9040846
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.17
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9027031
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.17
Output dim: 7, lower bound: -338.9059511, upper bound: 338.9055901
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.17
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -165.9965515, 132.0530243, -166.3823395, 132.3914948, -298.3880615, 298.4352417
1: -139.3771667, 117.2091446, -139.6709900, 117.4768524, -256.8540039, 256.8801270
2: -182.9610291, 118.6504669, -183.3416138, 118.8172531, -301.7782593, 301.9920654
3: -194.1609802, 102.6877899, -194.5800323, 102.9707870, -297.1317749, 297.2678223
4: -178.1408081, 136.2286987, -178.5437012, 136.5323639, -314.6731567, 314.7723999
5: -159.6695557, 124.1312103, -160.1475983, 124.4782486, -284.1477356, 284.2788086
6: -152.9830627, 147.3910065, -153.2779999, 147.7692566, -300.7523193, 300.6689758
7: -166.2316895, 140.2393494, -166.5496826, 140.5548553, -306.7864990, 306.7890320
8: -201.3135071, 137.6236572, -201.5581818, 137.9819794, -339.2954712, 339.1818237
9: -151.9037933, 149.1782074, -152.2440033, 149.5039062, -301.4077148, 301.4222107

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9053218, upper bound: 338.9047511
time: 10.66 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9053218, upper bound: 338.9047511
time: 10.77 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -170.8021393, 135.8643951, -163.6607513, 130.2377930, -301.0398865, 299.5251160
1: -143.3466644, 120.5318909, -137.3628998, 115.5498352, -258.8964539, 257.8947754
2: -188.2363892, 122.0633774, -180.3304138, 116.8786011, -305.1148987, 302.3937988
3: -199.7521667, 105.6213226, -191.3707581, 101.2878799, -301.0400391, 296.9920654
4: -183.2651825, 140.0964508, -175.5905762, 134.2784271, -317.5436096, 315.6870117
5: -164.2076721, 127.6427841, -157.5044861, 122.4271622, -286.6347656, 285.1472778
6: -157.3753510, 151.6097412, -150.7563934, 145.3503418, -302.7257080, 302.3661194
7: -171.0107574, 144.2021790, -163.7903137, 138.2445221, -309.2552795, 307.9924927
8: -207.1818390, 141.5537567, -198.2992706, 135.7566376, -342.9384766, 339.8530273
9: -156.2220154, 153.3689270, -149.7314148, 147.0339355, -303.2559509, 303.1003113

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9035341, upper bound: 338.9034488
time: 10.29 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9035341, upper bound: 338.9034488
time: 10.13 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -166.8089294, 132.6993408, -172.2980042, 137.0852814, -303.8942261, 304.9973145
1: -140.0791168, 117.8038025, -144.6846771, 121.7032166, -261.7822876, 262.4884644
2: -183.8847656, 119.2515335, -189.9389496, 123.0917358, -306.9765015, 309.1904297
3: -195.1363525, 103.1897888, -201.5921631, 106.6143265, -301.7506714, 304.7819214
4: -179.0258331, 136.9067383, -184.9445190, 141.4182739, -320.4440613, 321.8512268
5: -160.4478760, 124.7423706, -165.8235016, 128.9080505, -289.3559265, 290.5658264
6: -153.7501526, 148.1171112, -158.7758179, 153.0204010, -306.7705688, 306.8929443
7: -167.0877075, 140.9344330, -172.6132965, 145.5645905, -312.6522217, 313.5476990
8: -202.3357391, 138.2919769, -208.7986450, 142.8225250, -345.1582642, 347.0906067
9: -152.6607819, 149.9315796, -157.6873322, 154.8800659, -307.5408325, 307.6188965

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9053218, upper bound: 338.9062282
time: 11.66 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9053218, upper bound: 338.9062282
time: 13.08 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -171.5367737, 136.4495239, -169.5141296, 134.8824463, -306.4192200, 305.9636230
1: -143.9842529, 121.0704956, -142.3254395, 119.7309189, -263.7151794, 263.3959351
2: -189.0738373, 122.6058426, -186.8576050, 121.1075363, -310.1812744, 309.4634094
3: -200.6373444, 106.0756912, -198.3109131, 104.8930130, -305.5303650, 304.3865967
4: -184.0661163, 140.7091522, -181.9237823, 139.1119537, -323.1779785, 322.6329346
5: -164.9113922, 128.1941833, -163.1209869, 126.8088226, -291.7202148, 291.3150940
6: -158.0681763, 152.2654114, -156.1957092, 150.5447083, -308.6128845, 308.4611206
7: -171.7850952, 144.8314209, -169.7882996, 143.2000732, -314.9851685, 314.6197205
8: -208.1091156, 142.1573639, -205.4635773, 140.5462799, -348.6553650, 347.6208801
9: -156.9071503, 154.0513306, -155.1150208, 152.3530426, -309.2601318, 309.1663513

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9050195, upper bound: 338.9048543
time: 10.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9050195, upper bound: 338.9048543
time: 10.05 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -165.0199432, 131.2961731, -161.3927612, 128.4554901, -293.4754028, 292.6889343
1: -138.4903259, 116.4651489, -135.4620819, 113.9377747, -252.4281006, 251.9272156
2: -181.8432312, 117.9480591, -177.8349609, 115.2764435, -297.1196899, 295.7830200
3: -192.9868011, 102.1181946, -188.7057343, 99.9062881, -292.8930664, 290.8238831
4: -177.0444641, 135.3828125, -173.1585693, 132.4234619, -309.4679260, 308.5413513
5: -158.7119293, 123.3588562, -155.3448334, 120.7511902, -279.4631348, 278.7036743
6: -152.0659332, 146.5180511, -148.6736145, 143.3529816, -295.4188538, 295.1916504
7: -165.1954956, 139.4176025, -161.5151520, 136.3732147, -301.5687256, 300.9327087
8: -200.1798859, 136.7972260, -195.5634003, 133.8597717, -334.0396729, 332.3606262
9: -151.0035706, 148.2393188, -147.6943817, 144.9985046, -296.0020142, 295.9336243

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9027031
time: 9.78 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9027031
time: 11.57 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -170.0335388, 135.2663727, -158.7111816, 126.3324509, -296.3659668, 293.9775391
1: -142.6304474, 119.9335327, -133.1860352, 112.0384750, -254.6689148, 253.1195526
2: -187.3427582, 121.5070038, -174.8676758, 113.3654099, -300.7081604, 296.3746948
3: -198.8181152, 105.1787109, -185.5433807, 98.2486954, -297.0667419, 290.7221069
4: -182.3885651, 139.4198914, -170.2486877, 130.2026215, -312.5911255, 309.6685791
5: -163.4485779, 127.0233765, -152.7406921, 118.7299271, -282.1784973, 279.7640381
6: -156.6466980, 150.9188995, -146.1892395, 140.9692993, -297.6159668, 297.1080933
7: -170.1823120, 143.5532990, -158.7961273, 134.0959015, -304.2781372, 302.3493958
8: -206.2920074, 140.8920898, -192.3517761, 131.6669159, -337.9589233, 333.2438354
9: -155.5079498, 152.6124268, -145.2180481, 142.5639496, -298.0718384, 297.8304443

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9027031
time: 9.36 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9027031
time: 11.30 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -166.0124664, 132.0836487, -167.4692078, 133.2733002, -299.2857056, 299.5528564
1: -139.3470764, 117.1904449, -140.6113281, 118.2780838, -257.6251526, 257.8017578
2: -182.9683990, 118.6763000, -184.6084290, 119.6623993, -302.6307373, 303.2847290
3: -194.1770020, 102.7309723, -195.9100800, 103.6467438, -297.8237305, 298.6410522
4: -178.1254120, 136.2119751, -179.7308655, 137.4436188, -315.5690308, 315.9428406
5: -159.6586304, 124.1050034, -161.1688538, 125.2989349, -284.9575500, 285.2738342
6: -152.9977722, 147.4043121, -154.3202972, 148.7449036, -301.7426758, 301.7245483
7: -166.2393341, 140.2651062, -167.7406921, 141.5145721, -307.7538757, 308.0057983
8: -201.4195709, 137.6151581, -202.9951630, 138.8392639, -340.2588501, 340.6103210
9: -151.9307404, 149.1590118, -153.2861786, 150.5200500, -302.4507446, 302.4451294

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329
time: 7.84 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329
time: 10.98 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -170.9213715, 135.9721527, -164.7179413, 131.0954285, -302.0167847, 300.6900635
1: -143.3997955, 120.5842514, -138.2783203, 116.3290787, -259.7288513, 258.8625488
2: -188.3508148, 122.1573029, -181.5623779, 117.7007751, -306.0515747, 303.7196350
3: -199.8840637, 105.7278290, -192.6654816, 101.9458237, -301.8298950, 298.3932800
4: -183.3577576, 140.1626434, -176.7448273, 135.1643066, -318.5220642, 316.9074402
5: -164.2958221, 127.6896973, -158.4976959, 123.2238922, -287.5196533, 286.1873779
6: -157.4824524, 151.7125549, -151.7700500, 146.2981873, -303.7806396, 303.4825439
7: -171.1150665, 144.3123322, -164.9481201, 139.1769409, -310.2919617, 309.2603760
8: -207.4049225, 141.6264954, -199.6981354, 136.5890503, -343.9939575, 341.3246460
9: -156.3378754, 153.4364319, -150.7434998, 148.0213470, -304.3591919, 304.1798706

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9041329
time: 9.97 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329
time: 9.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.13 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9053218, upper bound: 338.9047511
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9053218, upper bound: 338.9047511
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9035341, upper bound: 338.9034488
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9035341, upper bound: 338.9034488
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9053218, upper bound: 338.9062282
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9053218, upper bound: 338.9062282
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9050195, upper bound: 338.9048543
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9050195, upper bound: 338.9048543
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9027031
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9027031
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9027031
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9027031
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9041329
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -165.9965515, 132.0530243, -159.0477905, 126.5746765, -292.5712280, 291.1007690
1: -139.3771667, 117.2091446, -133.5379333, 112.2659988, -251.6431580, 250.7470703
2: -182.9610291, 118.6504669, -175.2654419, 113.6256180, -296.5866089, 293.9158936
3: -194.1609802, 102.6877899, -185.9284668, 98.4391327, -292.6000977, 288.6162415
4: -178.1408081, 136.2286987, -170.6317749, 130.4969482, -308.6377563, 306.8604736
5: -159.6695557, 124.1312103, -153.0649414, 118.9783096, -278.6478271, 277.1961670
6: -152.9830627, 147.3910065, -146.5221100, 141.2702484, -294.2532959, 293.9131165
7: -166.2316895, 140.2393494, -159.1599731, 134.3921051, -300.6237793, 299.3993225
8: -201.3135071, 137.6236572, -192.7991486, 131.9246674, -333.2381287, 330.4227905
9: -151.9037933, 149.1782074, -145.5430756, 142.8740082, -294.7777100, 294.7212830

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9047525, upper bound: 338.9042696
time: 11.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9047525, upper bound: 338.9047511
time: 10.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -165.9965515, 132.0530243, -157.9628143, 125.7313690, -291.7279053, 290.0157776
1: -139.3771667, 117.2091446, -132.5550995, 111.4434662, -250.8206329, 249.7642517
2: -182.9610291, 118.6504669, -174.0249023, 112.8431702, -295.8041992, 292.6753540
3: -194.1609802, 102.6877899, -184.6233063, 97.8029480, -291.9639282, 287.3110352
4: -178.1408081, 136.2286987, -169.4144592, 129.5594025, -307.7001953, 305.6431580
5: -159.6695557, 124.1312103, -152.0036469, 118.1241226, -277.7936401, 276.1348572
6: -152.9830627, 147.3910065, -145.5032806, 140.3009796, -293.2840576, 292.8942566
7: -166.2316895, 140.2393494, -158.0111694, 133.4749603, -299.7066040, 298.2505188
8: -201.3135071, 137.6236572, -191.5286407, 131.0045776, -332.3180847, 329.1522827
9: -151.9037933, 149.1782074, -144.5401917, 141.8323669, -293.7361450, 293.7182922

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9047525, upper bound: 338.9042696
time: 12.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9047525, upper bound: 338.9047511
time: 10.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -170.8021393, 135.8643951, -156.3086395, 124.4069824, -295.2091064, 292.1730347
1: -143.3466644, 120.5318909, -131.2148590, 110.3259430, -253.6726074, 251.7467499
2: -188.2363892, 122.0633774, -172.2343445, 111.6746674, -299.9109802, 294.2977295
3: -199.7521667, 105.6213226, -182.6990662, 96.7445908, -296.4967041, 288.3203735
4: -183.2651825, 140.0964508, -167.6600189, 128.2287140, -311.4938965, 307.7563782
5: -164.2076721, 127.6427841, -150.4051971, 116.9137878, -281.1214600, 278.0479736
6: -157.3753510, 151.6097412, -143.9844055, 138.8357544, -296.2110901, 295.5941467
7: -171.0107574, 144.2021790, -156.3827057, 132.0663147, -303.0770264, 300.5848694
8: -207.1818390, 141.5537567, -189.5192719, 129.6855011, -336.8673401, 331.0730286
9: -156.2220154, 153.3689270, -143.0141449, 140.3878174, -296.6098328, 296.3830261

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8882065, upper bound: 338.8868741
time: 9.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8843762, upper bound: 338.8841739
time: 10.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -170.8021393, 135.8643951, -155.3121796, 123.6316299, -294.4337158, 291.1765747
1: -143.3466644, 120.5318909, -130.3045502, 109.5653305, -252.9119873, 250.8364410
2: -188.2363892, 122.0633774, -171.0902405, 110.9531250, -299.1893921, 293.1536255
3: -199.7521667, 105.6213226, -181.4969940, 96.1635437, -295.9157104, 287.1183167
4: -183.2651825, 140.0964508, -166.5373840, 127.3636703, -310.6288452, 306.6337280
5: -164.2076721, 127.6427841, -149.4289398, 116.1249313, -280.3326111, 277.0717163
6: -157.3753510, 151.6097412, -143.0463562, 137.9434357, -295.3187866, 294.6560669
7: -171.0107574, 144.2021790, -155.3216858, 131.2223206, -302.2330933, 299.5238647
8: -207.1818390, 141.5537567, -188.3518829, 128.8350677, -336.0169067, 329.9056396
9: -156.2220154, 153.3689270, -142.0906525, 139.4247284, -295.6467285, 295.4595337

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8882065, upper bound: 338.8868741
time: 12.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8843762, upper bound: 338.8841739
time: 10.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -166.8089294, 132.6993408, -164.8116150, 131.1473083, -297.9562378, 297.5109558
1: -140.0791168, 117.8038025, -138.4195709, 116.3805389, -256.4596558, 256.2233887
2: -183.8847656, 119.2515335, -181.6929474, 117.7904129, -301.6751709, 300.9444275
3: -195.1363525, 103.1897888, -192.7609100, 101.9880295, -297.1243896, 295.9506226
4: -179.0258331, 136.9067383, -176.8644409, 135.2548065, -314.2806091, 313.7711792
5: -160.4478760, 124.7423706, -158.5954437, 123.2924500, -283.7403259, 283.3377991
6: -153.7501526, 148.1171112, -151.8781281, 146.3838959, -300.1340332, 299.9952393
7: -167.0877075, 140.9344330, -165.0650177, 139.2704468, -306.3580933, 305.9994507
8: -202.3357391, 138.2919769, -199.8519135, 136.6371613, -338.9729004, 338.1438904
9: -152.6607819, 149.9315796, -150.8415680, 148.1102753, -300.7710571, 300.7731323

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8913923, upper bound: 338.8900567
time: 11.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8863644, upper bound: 338.8863603
time: 9.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -166.8089294, 132.6993408, -164.1298828, 130.6184845, -297.4274292, 296.8292236
1: -140.0791168, 117.8038025, -137.7825012, 115.8496323, -255.9287415, 255.5863037
2: -183.8847656, 119.2515335, -180.8995819, 117.2933426, -301.1780701, 300.1510315
3: -195.1363525, 103.1897888, -191.9367828, 101.5976028, -296.7339478, 295.1265564
4: -179.0258331, 136.9067383, -176.0870514, 134.6571503, -313.6829834, 312.9937134
5: -160.4478760, 124.7423706, -157.9129639, 122.7383423, -283.1862183, 282.6553345
6: -153.7501526, 148.1171112, -151.2293854, 145.7716064, -299.5217590, 299.3464661
7: -167.0877075, 140.9344330, -164.3294678, 138.6969299, -305.7845764, 305.2638855
8: -202.3357391, 138.2919769, -199.0671692, 136.0540009, -338.3897400, 337.3591309
9: -152.6607819, 149.9315796, -150.2177582, 147.4386444, -300.0994263, 300.1493530

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8913923, upper bound: 338.8900567
time: 10.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8863644, upper bound: 338.8863603
time: 10.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -171.5367737, 136.4495239, -162.0209961, 128.9385223, -300.4752808, 298.4704590
1: -143.9842529, 121.0704956, -136.0534821, 114.4027863, -258.3870239, 257.1239624
2: -189.0738373, 122.6058426, -178.6040039, 115.8005905, -304.8743591, 301.2098083
3: -200.6373444, 106.0756912, -189.4708710, 100.2620163, -300.8993225, 295.5465698
4: -184.0661163, 140.7091522, -173.8361816, 132.9423065, -317.0084229, 314.5452881
5: -164.9113922, 128.1941833, -155.8852844, 121.1875458, -286.0989075, 284.0794678
6: -158.0681763, 152.2654114, -149.2910614, 143.9016113, -301.9697876, 301.5564575
7: -171.7850952, 144.8314209, -162.2333069, 136.8996277, -308.6847229, 307.0647278
8: -208.1091156, 142.1573639, -196.5077515, 134.3548431, -342.4639587, 338.6651001
9: -156.9071503, 154.0513306, -148.2631531, 145.5757904, -302.4829407, 302.3144836

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8900520, upper bound: 338.8889740
time: 10.25 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8843762, upper bound: 338.8863453
time: 12.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -171.5367737, 136.4495239, -161.4145203, 128.4674377, -300.0041809, 297.8639832
1: -143.9842529, 121.0704956, -135.4786835, 113.9250336, -257.9093018, 256.5491943
2: -189.0738373, 122.6058426, -177.8927460, 115.3564301, -304.4302673, 300.4985962
3: -200.6373444, 106.0756912, -188.7340240, 99.9180908, -300.5554199, 294.8097229
4: -184.0661163, 140.7091522, -173.1389160, 132.4075165, -316.4735107, 313.8480835
5: -164.9113922, 128.1941833, -155.2756805, 120.6893463, -285.6006775, 283.4698486
6: -158.0681763, 152.2654114, -148.7113037, 143.3567657, -301.4249268, 300.9767151
7: -171.7850952, 144.8314209, -161.5725555, 136.3890839, -308.1741943, 306.4039917
8: -208.1091156, 142.1573639, -195.8109589, 133.8327332, -341.9418335, 337.9682922
9: -156.9071503, 154.0513306, -147.7073975, 144.9717560, -301.8789062, 301.7587280

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8900520, upper bound: 338.8889740
time: 11.16 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8843762, upper bound: 338.8863453
time: 9.89 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -165.0199432, 131.2961731, -158.4456635, 126.1196136, -291.1395569, 289.7418213
1: -138.4903259, 116.4651489, -132.9699554, 111.8502502, -250.3405762, 249.4350739
2: -181.8432312, 117.9480591, -174.5690765, 113.1800690, -295.0233154, 292.5171204
3: -192.9868011, 102.1181946, -185.2361603, 98.0801163, -291.0669250, 287.3543701
4: -177.0444641, 135.3828125, -169.9547729, 129.9779816, -307.0224609, 305.3374939
5: -158.7119293, 123.3588562, -152.4812317, 118.5275345, -277.2394714, 275.8400269
6: -152.0659332, 146.5180511, -145.9406586, 140.7269440, -292.7928162, 292.4587097
7: -165.1954956, 139.4176025, -158.5229645, 133.8652039, -299.0606689, 297.9405212
8: -200.1798859, 136.7972260, -192.0413666, 131.4384460, -331.6183472, 328.8385925
9: -151.0035706, 148.2393188, -144.9626770, 142.3180084, -293.3215332, 293.2019653

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8887185, upper bound: 338.8871682
time: 12.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8842111, upper bound: 338.8839934
time: 9.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -165.0199432, 131.2961731, -163.3973541, 130.0442963, -295.0642395, 294.6935425
1: -138.4903259, 116.4651489, -137.0540009, 115.2754822, -253.7657928, 253.5191193
2: -181.8432312, 117.9480591, -180.0024872, 116.6965103, -298.5397339, 297.9505615
3: -192.9868011, 102.1181946, -190.9972076, 101.1055527, -294.0923462, 293.1153259
4: -177.0444641, 135.3828125, -175.2340088, 133.9656067, -311.0100708, 310.6167908
5: -158.7119293, 123.3588562, -157.1616516, 122.1500549, -280.8619995, 280.5204773
6: -152.0659332, 146.5180511, -150.4671173, 145.0748138, -297.1407166, 296.9851685
7: -165.1954956, 139.4176025, -163.4487915, 137.9526367, -303.1481323, 302.8663330
8: -200.1798859, 136.7972260, -198.0790100, 135.4846191, -335.6644897, 334.8762207
9: -151.0035706, 148.2393188, -149.4157715, 146.6397858, -297.6433105, 297.6550293

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8887185, upper bound: 338.8871682
time: 10.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8842111, upper bound: 338.8839934
time: 9.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -170.0335388, 135.2663727, -156.3086395, 124.4069824, -294.4405212, 291.5750122
1: -142.6304474, 119.9335327, -131.2148590, 110.3259430, -252.9563904, 251.1483765
2: -187.3427582, 121.5070038, -172.2343445, 111.6746674, -299.0174255, 293.7413330
3: -198.8181152, 105.1787109, -182.6990662, 96.7445908, -295.5625916, 287.8777771
4: -182.3885651, 139.4198914, -167.6600189, 128.2287140, -310.6172180, 307.0798950
5: -163.4485779, 127.0233765, -150.4051971, 116.9137878, -280.3623657, 277.4285583
6: -156.6466980, 150.9188995, -143.9844055, 138.8357544, -295.4824219, 294.9032898
7: -170.1823120, 143.5532990, -156.3827057, 132.0663147, -302.2485657, 299.9359741
8: -206.2920074, 140.8920898, -189.5192719, 129.6855011, -335.9774780, 330.4113159
9: -155.5079498, 152.6124268, -143.0141449, 140.3878174, -295.8957520, 295.6265259

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874627, upper bound: 338.8862653
time: 10.29 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8841208, upper bound: 338.8838935
time: 10.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -170.0335388, 135.2663727, -155.3121796, 123.6316299, -293.6651306, 290.5785522
1: -142.6304474, 119.9335327, -130.3045502, 109.5653305, -252.1957703, 250.2380829
2: -187.3427582, 121.5070038, -171.0902405, 110.9531250, -298.2958374, 292.5972290
3: -198.8181152, 105.1787109, -181.4969940, 96.1635437, -294.9816589, 286.6757202
4: -182.3885651, 139.4198914, -166.5373840, 127.3636703, -309.7521973, 305.9572754
5: -163.4485779, 127.0233765, -149.4289398, 116.1249313, -279.5735168, 276.4523315
6: -156.6466980, 150.9188995, -143.0463562, 137.9434357, -294.5901489, 293.9652100
7: -170.1823120, 143.5532990, -155.3216858, 131.2223206, -301.4046326, 298.8750000
8: -206.2920074, 140.8920898, -188.3518829, 128.8350677, -335.1270752, 329.2439270
9: -155.5079498, 152.6124268, -142.0906525, 139.4247284, -294.9326782, 294.7030640

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874627, upper bound: 338.8862653
time: 12.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8841208, upper bound: 338.8838993
time: 10.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.32 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.9047525, upper bound: 338.9042696
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.9047525, upper bound: 338.9047511
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.9047525, upper bound: 338.9042696
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.9047525, upper bound: 338.9047511
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8882065, upper bound: 338.8868741
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8843762, upper bound: 338.8841739
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8882065, upper bound: 338.8868741
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8843762, upper bound: 338.8841739
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8913923, upper bound: 338.8900567
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8863644, upper bound: 338.8863603
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8913923, upper bound: 338.8900567
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8863644, upper bound: 338.8863603
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8900520, upper bound: 338.8889740
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8843762, upper bound: 338.8863453
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8900520, upper bound: 338.8889740
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8843762, upper bound: 338.8863453
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8887185, upper bound: 338.8871682
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8842111, upper bound: 338.8839934
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8887185, upper bound: 338.8871682
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8842111, upper bound: 338.8839934
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8874627, upper bound: 338.8862653
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8841208, upper bound: 338.8838935
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8874627, upper bound: 338.8862653
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 7, lower bound: -338.8841208, upper bound: 338.8838993
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.32
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.32
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.32
Output dim: 7, lower bound: -338.9026436, upper bound: 338.9041329
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.32
Output dim: 7, lower bound: -338.9041329, upper bound: 338.9041329
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=341.21295166015625
rel_dist={7: [-338.93400199054076, 338.93400199054076]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9231730, upper bound: 338.9230889
time: 12.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9228628, upper bound: 338.9228628
time: 11.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 24.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 24.57
Output dim: 7, lower bound: -338.9231730, upper bound: 338.9230889
IS_A2, status: Status.UNKNOWN, split count: 1, time: 24.57
Output dim: 7, lower bound: -338.9228628, upper bound: 338.9228628

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -175.0528107, 139.2155762, -177.9390411, 141.5046234, -316.5574341, 317.1545410
1: -147.0159912, 123.6377182, -149.4308777, 125.6884613, -272.7044678, 273.0686035
2: -192.9962158, 125.1462326, -196.1745300, 127.1894379, -320.1856079, 321.3207397
3: -204.8572540, 108.2642593, -208.2612762, 110.0477676, -314.9049988, 316.5255432
4: -187.9599152, 143.7206421, -191.0740967, 146.0958405, -334.0557556, 334.7947083
5: -168.3785706, 130.9101868, -171.1653595, 133.0740967, -301.4526062, 302.0755310
6: -161.3935852, 155.4163666, -164.0524597, 157.9739532, -319.3674927, 319.4687500
7: -175.4396667, 147.9055023, -178.3483429, 150.3311005, -325.7707520, 326.2538452
8: -212.2744598, 145.0615387, -215.7226868, 147.4452057, -359.7196655, 360.7841797
9: -160.2489624, 157.4055939, -162.8869171, 160.0151215, -320.2640686, 320.2924805

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9123290, upper bound: 338.9122168
time: 12.11 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9149599, upper bound: 338.9148519
time: 14.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -174.2116699, 138.5652008, -172.3863068, 137.1381989, -311.3498535, 310.9515076
1: -146.2442017, 122.9918976, -144.7220306, 121.7565765, -268.0007935, 267.7139282
2: -192.0296326, 124.5401611, -190.0398254, 123.2427979, -315.2723999, 314.5799866
3: -203.8439789, 107.7798386, -201.7372131, 106.6508408, -310.4948120, 309.5170593
4: -187.0102692, 142.9896851, -185.0800171, 141.5222321, -328.5325012, 328.0696411
5: -167.5493469, 130.2392883, -165.8263092, 128.9356537, -296.4849854, 296.0655823
6: -160.6019897, 154.6650085, -158.9267120, 153.0636139, -313.6655884, 313.5917358
7: -174.5427856, 147.1990662, -172.7429352, 145.6812897, -320.2240601, 319.9419861
8: -211.3038635, 144.3482513, -209.0324249, 142.8668976, -354.1707764, 353.3806763
9: -159.4755096, 156.5922089, -157.8307953, 155.0096588, -314.4851379, 314.4229431

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9120479, upper bound: 338.9119920
time: 13.58 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9120479, upper bound: 338.9145447
time: 13.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.04 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.04
Output dim: 7, lower bound: -338.9123290, upper bound: 338.9122168
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.04
Output dim: 7, lower bound: -338.9149599, upper bound: 338.9148519
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.04
Output dim: 7, lower bound: -338.9120479, upper bound: 338.9119920
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.04
Output dim: 7, lower bound: -338.9120479, upper bound: 338.9145447

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -162.9947052, 129.6917419, -161.8912659, 128.8295746, -291.8242798, 291.5830078
1: -136.8615875, 115.0697098, -135.9155579, 114.2859802, -251.1475525, 250.9852600
2: -179.6365662, 116.4661102, -178.3961029, 115.6384888, -295.2750549, 294.8621826
3: -190.5953674, 100.8616486, -189.2817993, 100.1956406, -290.7909851, 290.1434021
4: -174.9039459, 133.7576141, -173.6987000, 132.8366089, -307.7405396, 307.4562988
5: -156.8412170, 121.9201050, -155.8108521, 121.1101532, -277.9513550, 277.7309570
6: -150.1891479, 144.7583618, -149.1411896, 143.7895966, -293.9786682, 293.8995361
7: -163.1734619, 137.7238617, -162.0246887, 136.7813110, -299.9547729, 299.7484436
8: -197.5998077, 135.1633148, -196.1945801, 134.2725677, -331.8723755, 331.3578796
9: -149.1687927, 146.4572449, -148.1403809, 145.4441376, -294.6129150, 294.5975952

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9023557, upper bound: 338.9022016
time: 12.24 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9017578, upper bound: 338.9017470
time: 13.63 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -165.5249176, 131.6989136, -167.7192383, 133.4535828, -298.9784546, 299.4181213
1: -139.0243988, 116.9003677, -140.8531799, 118.4477310, -257.4720764, 257.7535400
2: -182.4839783, 118.3150482, -184.8955841, 119.8495407, -302.3335266, 303.2106018
3: -193.6155090, 102.4207687, -196.1901855, 103.7846527, -297.4001465, 298.6109619
4: -177.6522827, 135.8607941, -180.0024872, 137.6484833, -315.3007812, 315.8632812
5: -159.2625275, 123.8157959, -161.4023590, 125.4737091, -284.7361450, 285.2181396
6: -152.5616455, 147.0083160, -154.5574188, 148.9613190, -301.5229492, 301.5657043
7: -165.8014526, 139.8761597, -167.9965515, 141.7148895, -307.5162354, 307.8727112
8: -200.7405701, 137.2413177, -203.3268433, 139.0388336, -339.7794189, 340.5680847
9: -151.5133362, 148.7838440, -153.5003967, 150.7391510, -302.2525024, 302.2842407

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9038993, upper bound: 338.9037635
time: 14.82 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9033093, upper bound: 338.9032918
time: 13.66 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -161.9834290, 128.9064636, -156.1578827, 124.3196106, -286.3030090, 285.0643311
1: -135.9414368, 114.3004532, -131.0505981, 110.2223282, -246.1637421, 245.3510437
2: -178.4788971, 115.7365875, -172.0581970, 111.5599899, -290.0388489, 287.7947693
3: -189.3781586, 100.2703171, -182.5401611, 96.6877136, -286.0658569, 282.8104858
4: -173.7675629, 132.8824615, -167.5060425, 128.1106720, -301.8782349, 300.3884888
5: -155.8510437, 121.1207504, -150.2995453, 116.8365402, -272.6875916, 271.4202881
6: -149.2385864, 143.8546906, -143.8462219, 138.7169495, -287.9555359, 287.7009277
7: -162.1001892, 136.8707581, -156.2315979, 131.9771118, -294.0773010, 293.1023560
8: -196.4200897, 134.3055725, -189.2795410, 129.5411072, -325.9611816, 323.5850830
9: -148.2350922, 145.4841156, -142.9164429, 140.2696381, -288.5047302, 288.4005432

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9020979, upper bound: 338.9019614
time: 14.08 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9014420, upper bound: 338.9014825
time: 11.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -164.8278046, 131.1595612, -162.3024597, 129.1902161, -294.0180054, 293.4619751
1: -138.3748322, 116.3587494, -136.2586060, 114.6137772, -252.9886169, 252.6173401
2: -181.6758270, 117.8093338, -178.9074554, 115.9945679, -297.6703491, 296.7167969
3: -192.7729340, 102.0230789, -189.8265076, 100.4687347, -293.2416382, 291.8495789
4: -176.8588409, 135.2491608, -174.1526489, 133.1889038, -310.0476685, 309.4017944
5: -158.5668945, 123.2512665, -156.1878967, 121.4343033, -280.0011902, 279.4391479
6: -151.9002838, 146.3833923, -149.5544434, 144.1684418, -296.0687256, 295.9378052
7: -165.0527344, 139.2902679, -162.5271606, 137.1776276, -302.2303467, 301.8174438
8: -199.9387207, 136.6464539, -196.7940063, 134.5746307, -334.5133667, 333.4404602
9: -150.8745575, 148.1009216, -148.5715485, 145.8554688, -296.7299805, 296.6724243

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9036335, upper bound: 338.9035247
time: 13.83 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9029996, upper bound: 338.9029996
time: 11.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.62 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 7, lower bound: -338.9023557, upper bound: 338.9022016
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 7, lower bound: -338.9017578, upper bound: 338.9017470
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 7, lower bound: -338.9038993, upper bound: 338.9037635
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 7, lower bound: -338.9033093, upper bound: 338.9032918
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 7, lower bound: -338.9020979, upper bound: 338.9019614
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 7, lower bound: -338.9014420, upper bound: 338.9014825
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 7, lower bound: -338.9036335, upper bound: 338.9035247
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 7, lower bound: -338.9029996, upper bound: 338.9029996

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -160.1085358, 127.4028854, -160.1352081, 127.4369736, -287.5455017, 287.5380859
1: -134.4194489, 113.0257721, -134.4297638, 113.0419769, -247.4614258, 247.4555359
2: -176.4386444, 114.4123459, -176.4502258, 114.3886108, -290.8272705, 290.8625183
3: -187.1972504, 99.0736313, -187.2141266, 99.1075211, -286.3047791, 286.2877197
4: -171.7664032, 131.3637695, -171.7893677, 131.3797607, -303.1461792, 303.1530762
5: -154.0363464, 119.7420502, -154.1039734, 119.7849197, -273.8212585, 273.8460083
6: -147.5128784, 142.1876373, -147.5127869, 142.2249451, -289.7377625, 289.7003479
7: -160.2431946, 135.2678680, -160.2415314, 135.2867889, -295.5299377, 295.5093384
8: -194.1493988, 132.7915802, -194.0951385, 132.8293457, -326.9787598, 326.8866882
9: -146.4939880, 143.8318787, -146.5128174, 143.8466339, -290.3406372, 290.3446960

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8840610, upper bound: 338.8841850
time: 16.53 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8832354, upper bound: 338.8831398
time: 12.61 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -164.9439392, 131.2396698, -156.7981720, 124.7991638, -289.7430725, 288.0378418
1: -138.4123840, 116.3707199, -131.5949249, 110.6788101, -249.0911865, 247.9656372
2: -181.7476044, 117.8482437, -172.7612762, 112.0101776, -293.7577820, 290.6095276
3: -192.8246765, 102.0256424, -183.2767944, 97.0467606, -289.8714294, 285.3023682
4: -176.9229431, 135.2565002, -168.1727753, 128.6192017, -305.5421448, 303.4292603
5: -158.6042633, 123.2771225, -150.8648071, 117.2716293, -275.8758240, 274.1419373
6: -151.9337311, 146.4335785, -144.4224854, 139.2631531, -291.1968994, 290.8560791
7: -165.0537109, 139.2571869, -156.8618469, 132.4563293, -297.5100403, 296.1189575
8: -200.0550385, 136.7462311, -190.0949554, 130.1090698, -330.1640015, 326.8411560
9: -150.8407288, 148.0502167, -143.4386749, 140.8214264, -291.6621704, 291.4888916

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8841770, upper bound: 338.8844177
time: 14.65 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8832531, upper bound: 338.8831597
time: 14.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -162.6739197, 129.4379272, -165.9889526, 132.0814667, -294.7553101, 295.4268494
1: -136.6108398, 114.8812180, -139.3885498, 117.2219467, -253.8327637, 254.2697754
2: -179.3240204, 116.2873306, -182.9777985, 118.6186676, -297.9426270, 299.2651367
3: -190.2577667, 100.6539917, -194.1526489, 102.7119064, -292.9696655, 294.8066101
4: -174.5521393, 133.4958496, -178.1210175, 136.2133179, -310.7654419, 311.6168518
5: -156.4917603, 121.6647797, -159.7208405, 124.1685791, -280.6603088, 281.3856201
6: -149.9176941, 144.4682922, -152.9527893, 147.4195404, -297.3371887, 297.4210815
7: -162.9066925, 137.4504089, -166.2398987, 140.2426605, -303.1492920, 303.6902161
8: -197.3322601, 134.8982391, -201.2585144, 137.6167603, -334.9489746, 336.1567078
9: -148.8710785, 146.1902618, -151.8967285, 149.1654053, -298.0364075, 298.0869751

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8863473, upper bound: 338.8865359
time: 13.48 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8855318, upper bound: 338.8855238
time: 12.30 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -167.3607941, 133.1548462, -162.5128479, 129.3324738, -296.6932373, 295.6676941
1: -140.4795837, 118.1177521, -136.4380951, 114.7578583, -255.2374420, 254.5558472
2: -184.4660492, 119.6110229, -179.1327515, 116.1370544, -300.6030884, 298.7437744
3: -195.7103424, 103.5145416, -190.0536041, 100.5649643, -296.2752991, 293.5681152
4: -179.5479431, 137.2623901, -174.3537445, 133.3344269, -312.8823853, 311.6161499
5: -160.9152832, 125.0844727, -156.3473053, 121.5469513, -282.4622192, 281.4317627
6: -154.1964569, 148.5792542, -149.7312469, 144.3304596, -298.5268860, 298.3104858
7: -167.5603180, 141.3110809, -162.7141113, 137.2920837, -304.8524170, 304.0251770
8: -203.0538025, 138.7289734, -197.0874023, 134.7809601, -337.8347778, 335.8163452
9: -153.0771942, 150.2699890, -148.6898499, 146.0110168, -299.0881653, 298.9598389

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8864058, upper bound: 338.8866865
time: 12.40 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8855175, upper bound: 338.8855235
time: 11.42 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -159.0391693, 126.5723648, -154.3671722, 122.9002380, -281.9393616, 280.9395447
1: -133.4521484, 112.2147064, -129.5365906, 108.9537735, -242.4059143, 241.7512970
2: -175.2161865, 113.6420746, -170.0738525, 110.2863998, -285.5025635, 283.7159119
3: -185.9128876, 98.4461365, -180.4321594, 95.5780182, -281.4908752, 278.8782959
4: -170.5680389, 130.4391632, -165.5596008, 126.6246643, -297.1926880, 295.9987183
5: -152.9901581, 118.8994141, -148.5595551, 115.4852371, -268.4754028, 267.4589844
6: -146.5083008, 141.2314911, -142.1854553, 137.1213531, -283.6296387, 283.4169006
7: -159.1112061, 134.3657990, -154.4134827, 130.4535065, -289.5646667, 288.7792969
8: -192.9012909, 131.8856964, -187.1394806, 128.0696106, -320.9708862, 319.0251770
9: -145.5065460, 142.8059845, -141.2567444, 138.6407928, -284.1473389, 284.0626831

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8839903, upper bound: 338.8842020
time: 14.03 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8831836, upper bound: 338.8830921
time: 13.50 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -164.0949860, 130.5756073, -151.1969604, 120.3911133, -284.4860535, 281.7725220
1: -137.6255493, 115.7112808, -126.8373642, 106.7071228, -244.3326569, 242.5486450
2: -180.7625275, 117.2309418, -166.5660095, 108.0235291, -288.7860718, 283.7969360
3: -191.7924500, 101.5318756, -176.6877289, 93.6212006, -285.4136353, 278.2196045
4: -175.9576569, 134.5102692, -162.1203918, 124.0013123, -299.9589539, 296.6306763
5: -157.7680511, 122.5953064, -145.4810486, 113.0967331, -270.8647766, 268.0763245
6: -151.1274872, 145.6683960, -139.2480316, 134.3051605, -285.4326477, 284.9163818
7: -164.1400604, 138.5363617, -151.1988678, 127.7620163, -291.9020691, 289.7351990
8: -199.0634766, 136.0133972, -183.3341370, 125.4825439, -324.5459900, 319.3475342
9: -150.0489960, 147.2170410, -138.3330994, 135.7637634, -285.8127441, 285.5500793

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8840799, upper bound: 338.8843756
time: 13.60 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8831692, upper bound: 338.8830846
time: 12.90 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -161.9304657, 128.8619995, -160.5439758, 127.7961578, -289.7266235, 289.4059143
1: -135.9237976, 114.3058624, -134.7708130, 113.3680801, -249.2918701, 249.0766754
2: -178.4649963, 115.7486801, -176.9588013, 114.7443008, -293.2092896, 292.7074890
3: -189.3621979, 100.2266006, -187.7561035, 99.3784790, -288.7406311, 287.9826660
4: -173.7098541, 132.8452606, -172.2412720, 131.7299500, -305.4397888, 305.0865173
5: -155.7506409, 121.0656433, -154.4787903, 120.1078415, -275.8584900, 275.5444336
6: -149.2135773, 143.8013611, -147.9243164, 142.6013336, -291.8149109, 291.7256775
7: -162.1114349, 136.8256073, -160.7423401, 135.6817322, -297.7931213, 297.5678406
8: -196.4770966, 134.2646332, -194.6928711, 133.1298218, -329.6069336, 328.9575195
9: -148.1902161, 145.4649048, -146.9423218, 144.2561340, -292.4462891, 292.4071350

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8839903, upper bound: 338.8865225
time: 14.77 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8855399, upper bound: 338.8855417
time: 11.96 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -166.7946472, 132.7147522, -157.2142334, 125.1603851, -291.9550171, 289.9289551
1: -139.9392700, 117.6667252, -131.9410400, 111.0071335, -250.9464111, 249.6077576
2: -183.7975769, 119.1970825, -173.2742004, 112.3647156, -296.1622925, 292.4712219
3: -195.0149689, 103.1964645, -183.8241882, 97.3235474, -292.3385010, 287.0206299
4: -178.8924866, 136.7582245, -168.6295776, 128.9720001, -307.8644714, 305.3878174
5: -160.3446503, 124.6154938, -151.2465973, 117.5957413, -277.9403992, 275.8620911
6: -153.6554260, 148.0704498, -144.8365021, 139.6437378, -293.2991638, 292.9069519
7: -166.9402924, 140.8336792, -157.3625793, 132.8537292, -299.7940063, 298.1962585
8: -202.4080963, 138.2391968, -190.6940765, 130.4118195, -332.8199158, 328.9332886
9: -152.5544434, 149.7012634, -143.8680267, 141.2330170, -293.7874451, 293.5692749

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8863403, upper bound: 338.8866631
time: 14.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8855017, upper bound: 338.8855017
time: 10.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.72 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8840610, upper bound: 338.8841850
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8832354, upper bound: 338.8831398
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8841770, upper bound: 338.8844177
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8832531, upper bound: 338.8831597
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8863473, upper bound: 338.8865359
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8855318, upper bound: 338.8855238
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8864058, upper bound: 338.8866865
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8855175, upper bound: 338.8855235
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8839903, upper bound: 338.8842020
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8831836, upper bound: 338.8830921
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8840799, upper bound: 338.8843756
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8831692, upper bound: 338.8830846
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8839903, upper bound: 338.8865225
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8855399, upper bound: 338.8855417
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8863403, upper bound: 338.8866631
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 26.72
Output dim: 7, lower bound: -338.8855017, upper bound: 338.8855017

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -166.2819214, 132.3040466, -160.1309204, 127.4526596, -293.7345581, 292.4349365
1: -139.5686493, 117.3494568, -134.4262848, 113.0611954, -252.6298523, 251.7757416
2: -183.2713470, 118.8496094, -176.4943848, 114.4554214, -297.7267761, 295.3439331
3: -194.4518127, 102.8446960, -187.2736359, 99.0862579, -293.5380859, 290.1183472
4: -178.3847809, 136.3744202, -171.7857819, 131.3732452, -309.7580261, 308.1601868
5: -159.8786163, 124.2774887, -154.0577545, 119.7660980, -279.6447144, 278.3351746
6: -153.2076721, 147.6171570, -147.5470581, 142.2064362, -295.4141235, 295.1642151
7: -166.4715576, 140.3997803, -160.3095551, 135.2798462, -301.7514038, 300.7092285
8: -201.7630463, 137.8354034, -194.2373199, 132.8072205, -334.5702515, 332.0726929
9: -152.0882263, 149.2931213, -146.5072937, 143.8542023, -295.9424438, 295.8004150

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8755884, upper bound: 338.8757995
time: 13.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8747336, upper bound: 338.8750338
time: 14.21 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -165.6752625, 131.8320007, -154.7515411, 123.2169952, -288.8922424, 286.5835571
1: -138.9945984, 116.8693542, -129.8629913, 109.2529755, -248.2475586, 246.7323456
2: -182.5579834, 118.4073944, -170.5459137, 110.6268921, -293.1848755, 288.9533081
3: -193.7095184, 102.5010147, -180.9510651, 95.7940216, -289.5034790, 283.4520874
4: -177.6860046, 135.8368683, -165.9749603, 126.9450073, -304.6310120, 301.8118286
5: -159.2689667, 123.7781982, -148.8793640, 115.7541275, -275.0231018, 272.6575623
6: -152.6299744, 147.0720062, -142.5794983, 137.4472198, -290.0772095, 289.6514893
7: -165.8110809, 139.8881836, -154.8776550, 130.7737732, -296.5848389, 294.7658081
8: -201.0693359, 137.3116150, -187.7479706, 128.3708038, -329.4401245, 325.0595703
9: -151.5288544, 148.6878357, -141.6118622, 139.0031128, -290.5319519, 290.2996826

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8758860, upper bound: 338.8760942
time: 12.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8750460, upper bound: 338.8753781
time: 12.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.11 seconds
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.11
Output dim: 7, lower bound: -338.8755884, upper bound: 338.8757995
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.11
Output dim: 7, lower bound: -338.8747336, upper bound: 338.8750338
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.11
Output dim: 7, lower bound: -338.8758860, upper bound: 338.8760942
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.11
Output dim: 7, lower bound: -338.8750460, upper bound: 338.8753781
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=341.21295166015625
rel_dist={7: [-338.93322554933513, 338.933225507835]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9241222, upper bound: 338.9239559
time: 11.22 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9234949, upper bound: 338.9234949
time: 10.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.61
Output dim: 7, lower bound: -338.9241222, upper bound: 338.9239559
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.61
Output dim: 7, lower bound: -338.9234949, upper bound: 338.9234949

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -175.0528107, 139.2155762, -180.3886108, 143.4473267, -318.5001221, 319.6041260
1: -147.0159912, 123.6377182, -151.4808044, 127.4288788, -274.4448853, 275.1184998
2: -192.9962158, 125.1462326, -198.8718109, 128.9236145, -321.9198303, 324.0180359
3: -204.8572540, 108.2642593, -211.1506805, 111.5615768, -316.4188232, 319.4149475
4: -187.9599152, 143.7206421, -193.7170410, 148.1121826, -336.0720825, 337.4376526
5: -168.3785706, 130.9101868, -173.5310059, 134.9103851, -303.2889099, 304.4411011
6: -161.3935852, 155.4163666, -166.3090057, 160.1446991, -321.5382690, 321.7253113
7: -175.4396667, 147.9055023, -180.8169250, 152.3898010, -327.8294678, 328.7224121
8: -212.2744598, 145.0615387, -218.6491852, 149.4682617, -361.7427368, 363.7106934
9: -160.2489624, 157.4055939, -165.1253662, 162.2294769, -322.4784241, 322.5309448

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9134574, upper bound: 338.9132579
time: 11.06 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9134574, upper bound: 338.9157945
time: 10.78 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -174.2116699, 138.5652008, -175.3473206, 139.4766693, -313.6882935, 313.9124756
1: -146.2442017, 122.9918976, -147.2184143, 123.8569489, -270.1011353, 270.2103271
2: -192.0296326, 124.5401611, -193.3067322, 125.3433685, -317.3729858, 317.8468933
3: -203.8439789, 107.7798386, -205.2236633, 108.4712677, -312.3152466, 313.0035095
4: -187.0102692, 142.9896851, -188.2770691, 143.9616089, -330.9718628, 331.2667236
5: -167.5493469, 130.2392883, -168.6805267, 131.1485748, -298.6979370, 298.9197388
6: -160.6019897, 154.6650085, -161.6573792, 155.6851654, -316.2871399, 316.3223877
7: -174.5427856, 147.1990662, -175.7310028, 148.1661224, -322.7089233, 322.9300537
8: -211.3038635, 144.3482513, -212.5859985, 145.3090057, -356.6128540, 356.9342651
9: -159.4755096, 156.5922089, -160.5321808, 157.6826019, -317.1580811, 317.1243896

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9128825, upper bound: 338.9127934
time: 10.82 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9151810, upper bound: 338.9151809
time: 10.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 7, lower bound: -338.9134574, upper bound: 338.9132579
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 7, lower bound: -338.9134574, upper bound: 338.9157945
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 7, lower bound: -338.9128825, upper bound: 338.9127934
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 7, lower bound: -338.9151810, upper bound: 338.9151809

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -166.2113495, 132.2317810, -164.3070526, 130.7453003, -296.9566650, 296.5387573
1: -139.5699158, 117.3549118, -137.9356689, 116.0022507, -255.5721741, 255.2905884
2: -183.1992035, 118.7808533, -181.0562744, 117.3483810, -300.5475769, 299.8370667
3: -194.3993988, 102.8359070, -192.1316528, 101.6882553, -296.0876160, 294.9675598
4: -178.3863525, 136.4149017, -176.3048706, 134.8244934, -313.2108459, 312.7197266
5: -159.9184265, 124.3176651, -158.1434784, 122.9217224, -282.8401489, 282.4611511
6: -153.1774292, 147.6010284, -151.3666534, 145.9301758, -299.1076050, 298.9676819
7: -166.4445190, 140.4392700, -164.4589386, 138.8109894, -305.2554932, 304.8981323
8: -201.5130768, 137.8027191, -199.0796661, 136.2679596, -337.7810364, 336.8823853
9: -152.1240540, 149.3775940, -150.3477631, 147.6279755, -299.7520142, 299.7253418

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9038674, upper bound: 338.9035475
time: 12.61 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026822, upper bound: 338.9026507
time: 12.58 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -167.8209686, 133.5098114, -170.1831055, 135.4078827, -303.2288513, 303.6928406
1: -140.9499969, 118.5226517, -142.9152069, 120.1995087, -261.1495056, 261.4378357
2: -185.0160980, 119.9606705, -187.6096039, 121.5942841, -306.6103821, 307.5702515
3: -196.3238373, 103.8282089, -199.0966492, 105.3071060, -301.6309204, 302.9248352
4: -180.1360474, 137.7544098, -182.6619568, 139.6769562, -319.8129883, 320.4162598
5: -161.4591827, 125.5246811, -163.7813416, 127.3219070, -288.7810974, 289.3059998
6: -154.6891479, 149.0343323, -156.8273163, 151.1456299, -305.8347778, 305.8616333
7: -168.1226501, 141.8104401, -170.4811249, 143.7863159, -311.9089355, 312.2915649
8: -203.5182190, 139.1253052, -206.2715149, 141.0748444, -344.5929565, 345.3967590
9: -153.6176910, 150.8608398, -155.7534180, 152.9673462, -306.5850220, 306.6142273

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9053391, upper bound: 338.9050447
time: 12.00 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9041793, upper bound: 338.9041237
time: 11.29 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -165.2514801, 131.4875031, -159.1425629, 126.6777725, -291.9292297, 290.6300659
1: -138.6945648, 116.6229706, -133.5663147, 112.3406372, -251.0352020, 250.1892700
2: -182.0997009, 118.0888596, -175.3525085, 113.6795807, -295.7792969, 293.4413757
3: -193.2435913, 102.2764053, -186.0555420, 98.5227737, -291.7663574, 288.3319092
4: -177.3062592, 135.5836334, -170.7295837, 130.5695953, -307.8758545, 306.3132324
5: -158.9777374, 123.5572128, -153.1764526, 119.0684204, -278.0461426, 276.7336426
6: -152.2752380, 146.7431335, -146.5985107, 141.3602905, -293.6354980, 293.3415833
7: -165.4243011, 139.6310730, -159.2451782, 134.4842987, -299.9085999, 298.8762207
8: -200.3967438, 136.9890289, -192.8630676, 132.0026245, -332.3993530, 329.8521118
9: -151.2385864, 148.4526215, -145.6405640, 142.9659882, -294.2045898, 294.0931702

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9033657, upper bound: 338.9030859
time: 10.18 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9020722, upper bound: 338.9021385
time: 11.13 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -167.0963593, 132.9503479, -165.2460632, 131.5168304, -298.6131897, 298.1963501
1: -140.2776184, 117.9614944, -138.7393494, 116.7015076, -256.9791260, 256.7008057
2: -184.1782837, 119.4361649, -182.1557159, 118.0847931, -302.2630005, 301.5918884
3: -195.4482880, 103.4147263, -193.2923431, 102.2789917, -297.7272644, 296.7070618
4: -179.3127594, 137.1199799, -177.3312378, 135.6131287, -314.9258423, 314.4512329
5: -160.7390594, 124.9404602, -159.0257111, 123.6358490, -284.3748474, 283.9661560
6: -154.0034943, 148.3857574, -152.2703857, 146.7757263, -300.7792053, 300.6560974
7: -167.3463898, 141.2018585, -165.4978180, 139.6483765, -306.9947510, 306.6996765
8: -202.6854095, 138.5080872, -200.3274078, 137.0050812, -339.6904297, 338.8355103
9: -152.9532013, 150.1536560, -151.2581635, 148.5127716, -301.4659729, 301.4118042

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9048142, upper bound: 338.9045960
time: 12.28 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9035845, upper bound: 338.9035845
time: 10.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.46 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.46
Output dim: 7, lower bound: -338.9038674, upper bound: 338.9035475
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.46
Output dim: 7, lower bound: -338.9026822, upper bound: 338.9026507
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.46
Output dim: 7, lower bound: -338.9053391, upper bound: 338.9050447
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.46
Output dim: 7, lower bound: -338.9041793, upper bound: 338.9041237
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.46
Output dim: 7, lower bound: -338.9033657, upper bound: 338.9030859
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.46
Output dim: 7, lower bound: -338.9020722, upper bound: 338.9021385
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.46
Output dim: 7, lower bound: -338.9048142, upper bound: 338.9045960
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.46
Output dim: 7, lower bound: -338.9035845, upper bound: 338.9035845

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -163.3269501, 129.9447327, -163.5219879, 130.1227722, -293.4497070, 293.4667053
1: -137.1294098, 115.3124847, -137.2713623, 115.4461594, -252.5755615, 252.5838470
2: -180.0038757, 116.7289734, -180.1862946, 116.7897644, -296.7936096, 296.9152527
3: -191.0037842, 101.0493622, -191.2070160, 101.2017746, -292.2055054, 292.2563171
4: -175.2508698, 134.0229645, -175.4512939, 134.1729889, -309.4238586, 309.4742432
5: -157.1156158, 122.1411743, -157.3805695, 122.3293304, -279.4449463, 279.5217285
6: -150.5031891, 145.0317078, -150.6385651, 145.2306061, -295.7337952, 295.6701965
7: -163.5165710, 137.9853668, -163.6616364, 138.1428223, -301.6593933, 301.6469116
8: -198.0653229, 135.4327850, -198.1411285, 135.6228638, -333.6881104, 333.5739136
9: -149.4510956, 146.7543640, -149.6201324, 146.9136810, -296.3647766, 296.3744812

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026822, upper bound: 338.9026507
time: 10.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026822, upper bound: 338.9026507
time: 11.52 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -168.1471558, 133.7682953, -160.6267090, 127.8328400, -295.9799805, 294.3950195
1: -141.1105957, 118.6459274, -134.8142700, 113.3958740, -254.5064697, 253.4602051
2: -185.2954254, 120.1529312, -176.9844666, 114.7266617, -300.0220642, 297.1373901
3: -196.6125793, 103.9916611, -187.7920990, 99.4127045, -296.0252686, 291.7837524
4: -180.3906250, 137.9030151, -172.3112946, 131.7768555, -312.1674805, 310.2142944
5: -161.6680145, 125.6641388, -154.5691223, 120.1479340, -281.8159485, 280.2332458
6: -154.9090118, 149.2639313, -147.9566040, 142.6592407, -297.5681763, 297.2205200
7: -168.3108063, 141.9608154, -160.7277374, 135.6861267, -303.9969177, 302.6884766
8: -203.9517212, 139.3747559, -194.6724091, 133.2588806, -337.2106018, 334.0471191
9: -153.7831573, 150.9584045, -146.9499817, 144.2875519, -298.0707092, 297.9083862

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026822, upper bound: 338.9026507
time: 10.57 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9026822, upper bound: 338.9026507
time: 10.78 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -164.9625397, 131.2430420, -169.4105377, 134.7952576, -299.7577820, 300.6535645
1: -138.5303802, 116.4985123, -142.2612457, 119.6522751, -258.1826477, 258.7597351
2: -181.8481903, 117.9277649, -186.7532959, 121.0446777, -302.8928833, 304.6809998
3: -192.9579315, 102.0571747, -198.1868134, 104.8281250, -297.7860718, 300.2439880
4: -177.0283661, 135.3835907, -181.8218994, 139.0360565, -316.0643616, 317.2055054
5: -158.6813965, 123.3680267, -163.0303497, 126.7391586, -285.4204712, 286.3983765
6: -152.0389252, 146.4878845, -156.1109619, 150.4573059, -302.4961853, 302.5988464
7: -165.2204132, 139.3786621, -169.6967773, 143.1289520, -308.3493652, 309.0754395
8: -200.1013794, 136.7763824, -205.3478851, 140.4400024, -340.5413818, 342.1242676
9: -150.9684143, 148.2609253, -155.0375061, 152.2646179, -303.2330322, 303.2984009

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9041793, upper bound: 338.9041237
time: 12.39 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9041793, upper bound: 338.9041237
time: 10.38 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -169.6725464, 134.9790039, -166.4183807, 132.4286346, -302.1011047, 301.3973389
1: -142.4199829, 119.7520218, -139.7238312, 117.5320435, -259.9520264, 259.4758606
2: -187.0166931, 121.2688217, -183.4424744, 118.9107590, -305.9274292, 304.7113037
3: -198.4376678, 104.9323425, -194.6596375, 102.9793701, -301.4170227, 299.5919800
4: -182.0488739, 139.1704559, -178.5773926, 136.5577240, -318.6065979, 317.7478638
5: -163.1277313, 126.8060074, -160.1265259, 124.4824753, -287.6101990, 286.9325256
6: -156.3396606, 150.6199646, -153.3382568, 147.7973175, -304.1369324, 303.9581604
7: -169.8987885, 143.2599945, -166.6611786, 140.5883179, -310.4871216, 309.9211731
8: -205.8523254, 140.6269684, -201.7606049, 137.9961548, -343.8484802, 342.3875732
9: -155.1974030, 152.3633881, -152.2749176, 149.5492706, -304.7466736, 304.6382751

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9041793, upper bound: 338.9041237
time: 10.78 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9041793, upper bound: 338.9041237
time: 10.49 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -162.3107147, 129.1562958, -158.3461609, 126.0466003, -288.3572998, 287.5024109
1: -136.2079468, 114.5396347, -132.8929138, 111.7765274, -247.9844666, 247.4325562
2: -178.8409271, 115.9971466, -174.4699097, 113.1132126, -291.9541321, 290.4670410
3: -189.7822266, 100.4544525, -185.1180420, 98.0292816, -287.8114624, 285.5724487
4: -174.1104889, 133.1433868, -169.8638916, 129.9086761, -304.0191650, 303.0072327
5: -156.1202393, 121.3385239, -152.4025574, 118.4675598, -274.5877380, 273.7410889
6: -149.5482025, 144.1231079, -145.8599548, 140.6507416, -290.1989441, 289.9830627
7: -162.4391022, 137.1290283, -158.4365082, 133.8066406, -296.2457275, 295.5655518
8: -196.8823242, 134.5721436, -191.9113770, 131.3482361, -328.2305603, 326.4834595
9: -148.5132751, 145.7779388, -144.9024658, 142.2416229, -290.7548828, 290.6803894

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9020722, upper bound: 338.9021385
time: 11.23 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9020722, upper bound: 338.9021385
time: 10.59 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -167.3447266, 133.1424103, -155.5333557, 123.8197403, -291.1644287, 288.6757202
1: -140.3640289, 118.0216522, -130.5021515, 109.7838211, -250.1478271, 248.5238037
2: -184.3632965, 119.5707245, -171.3577881, 111.1066208, -295.4699097, 290.9285278
3: -195.6368408, 103.5273285, -181.7983398, 96.2915649, -291.9284058, 285.3256226
4: -179.4768066, 137.1967773, -166.8119202, 127.5801544, -307.0569153, 304.0086975
5: -160.8764496, 125.0183029, -149.6707306, 116.3475418, -277.2239990, 274.6890259
6: -154.1476746, 148.5415039, -143.2533264, 138.1514435, -292.2991333, 291.7948303
7: -167.4463501, 141.2815399, -155.5843506, 131.4179840, -298.8643188, 296.8659058
8: -203.0188293, 138.6828613, -188.5386353, 129.0506287, -332.0694580, 327.2214966
9: -153.0361023, 150.1694641, -142.3068542, 139.6881714, -292.7242432, 292.4763184

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8853742, upper bound: 338.8859962
time: 10.67 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8837202, upper bound: 338.8835457
time: 10.12 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -164.1919708, 130.6471863, -164.4652557, 130.8977814, -295.0897522, 295.1124268
1: -137.8208008, 115.9038696, -138.0786896, 116.1483459, -253.9691467, 253.9825439
2: -180.9602356, 117.3706589, -181.2905884, 117.5296249, -298.4898376, 298.6612549
3: -192.0294647, 101.6141815, -192.3729401, 101.7948456, -293.8243103, 293.9871216
4: -176.1564484, 134.7104645, -176.4824066, 134.9653168, -311.1217651, 311.1928711
5: -157.9161072, 122.7496719, -158.2667694, 123.0468674, -280.9629211, 281.0164490
6: -151.3102417, 145.7977295, -151.5463104, 146.0800171, -297.3902283, 297.3439941
7: -164.3985901, 138.7312927, -164.7054138, 138.9841156, -303.3826904, 303.4367065
8: -199.2152557, 136.1212311, -199.3943787, 136.3636017, -335.5787659, 335.5155945
9: -150.2625885, 147.5118256, -150.5348053, 147.8027191, -298.0652466, 298.0465698

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9035845, upper bound: 338.9035845
time: 10.38 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.9035845, upper bound: 338.9035845
time: 10.08 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -169.0815430, 134.5198364, -161.5461426, 128.5872040, -297.6687317, 296.0659790
1: -141.8565979, 119.2831345, -135.6008759, 114.0801468, -255.9367218, 254.8840027
2: -186.3204956, 120.8369980, -178.0597534, 115.4460068, -301.7665100, 298.8967590
3: -197.7128906, 104.5990067, -188.9289093, 99.9915237, -297.7044067, 293.5279236
4: -181.3670044, 138.6445923, -173.3150024, 132.5476074, -313.9146118, 311.9595947
5: -162.5343933, 126.3188782, -155.4331818, 120.8450317, -283.3794250, 281.7520447
6: -155.7759705, 150.0887451, -148.8406219, 143.4853210, -299.2612915, 298.9293213
7: -169.2534180, 142.7611694, -161.7427979, 136.5050354, -305.7584534, 304.5039673
8: -205.1768188, 140.1161499, -195.8929901, 133.9784698, -339.1552734, 336.0091553
9: -154.6509399, 151.7708588, -147.8383636, 145.1526642, -299.8035889, 299.6092224

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8874235, upper bound: 338.8881357
time: 10.78 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8858617, upper bound: 338.8858617
time: 10.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.37 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9026822, upper bound: 338.9026507
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9026822, upper bound: 338.9026507
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9026822, upper bound: 338.9026507
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9026822, upper bound: 338.9026507
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9041793, upper bound: 338.9041237
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9041793, upper bound: 338.9041237
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9041793, upper bound: 338.9041237
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9041793, upper bound: 338.9041237
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9020722, upper bound: 338.9021385
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9020722, upper bound: 338.9021385
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.8853742, upper bound: 338.8859962
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.8837202, upper bound: 338.8835457
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9035845, upper bound: 338.9035845
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.9035845, upper bound: 338.9035845
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.8874235, upper bound: 338.8881357
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 22.37
Output dim: 7, lower bound: -338.8858617, upper bound: 338.8858617

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -163.3269501, 129.9447327, -161.3970184, 128.4376068, -291.7645264, 291.3417358
1: -137.1294098, 115.3124847, -135.4730377, 113.9407120, -251.0701141, 250.7855072
2: -180.0038757, 116.7289734, -177.8313904, 115.2772446, -295.2810364, 294.5603638
3: -191.0037842, 101.0493622, -188.7047424, 99.8850250, -290.8887329, 289.7540588
4: -175.2508698, 134.0229645, -173.1408539, 132.4098511, -307.6607056, 307.1637878
5: -157.1156158, 122.1411743, -155.3151855, 120.7257080, -277.8413086, 277.4563599
6: -150.5031891, 145.0317078, -148.6680145, 143.3370819, -293.8401794, 293.6996460
7: -163.5165710, 137.9853668, -161.5036163, 136.3343048, -299.8508911, 299.4888611
8: -198.0653229, 135.4327850, -195.6003876, 133.8764038, -331.9417114, 331.0331726
9: -149.4510956, 146.7543640, -147.6504822, 144.9803162, -294.4313965, 294.4048462

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8872254, upper bound: 338.8862086
time: 12.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8839052, upper bound: 338.8837217
time: 10.27 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -163.3269501, 129.9447327, -166.2586212, 132.2958374, -295.6228027, 296.2033691
1: -137.1294098, 115.3124847, -139.4870758, 117.3055878, -254.4349976, 254.7995605
2: -180.0038757, 116.7289734, -183.1687164, 118.7324600, -298.7363281, 299.8976746
3: -191.0037842, 101.0493622, -194.3628845, 102.8555527, -293.8592834, 295.4122314
4: -175.2508698, 134.0229645, -178.3255005, 136.3253326, -311.5762024, 312.3484497
5: -157.1156158, 122.1411743, -159.9100189, 124.2825546, -281.3981628, 282.0511475
6: -150.5031891, 145.0317078, -153.1117554, 147.6081085, -298.1112976, 298.1433716
7: -163.5165710, 137.9853668, -166.3413086, 140.3481750, -303.8647461, 304.3266296
8: -198.0653229, 135.4327850, -201.5354614, 137.8514252, -335.9166870, 336.9682617
9: -149.4510956, 146.7543640, -152.0230408, 149.2240448, -298.6751099, 298.7774048

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8872254, upper bound: 338.8862086
time: 12.23 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8839052, upper bound: 338.8837217
time: 10.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -168.1471558, 133.7682953, -155.3496552, 123.6480942, -291.7952271, 289.1179504
1: -141.1105957, 118.6459274, -130.4012299, 109.6467514, -250.7573547, 249.0471497
2: -185.2954254, 120.1529312, -171.1735535, 110.9915771, -296.2869873, 291.3264771
3: -196.6125793, 103.9916611, -181.5685272, 96.1517258, -292.7642822, 285.5601501
4: -180.3906250, 137.9030151, -166.6196899, 127.4346542, -307.8252869, 304.5227051
5: -161.6680145, 125.6641388, -149.4740448, 116.1910629, -277.8590698, 275.1381836
6: -154.9090118, 149.2639313, -143.0963135, 137.9834900, -292.8924561, 292.3602295
7: -168.3108063, 141.9608154, -155.4107208, 131.2520294, -299.5628357, 297.3715210
8: -203.9517212, 139.3747559, -188.3708038, 128.9017181, -332.8534546, 327.7455444
9: -153.7831573, 150.9584045, -142.1288910, 139.5174103, -293.3005676, 293.0872803

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8865759, upper bound: 338.8856684
time: 11.27 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8839100, upper bound: 338.8837365
time: 11.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -168.1471558, 133.7682953, -154.3829346, 122.8954163, -291.0425720, 288.1512451
1: -141.1105957, 118.6459274, -129.5153656, 108.9066544, -250.0172424, 248.1612854
2: -185.2954254, 120.1529312, -170.0613861, 110.2904129, -295.5858154, 290.2142944
3: -196.6125793, 103.9916611, -180.4007874, 95.5890045, -292.2015381, 284.3924255
4: -180.3906250, 137.9030151, -165.5285797, 126.5938797, -306.9844971, 303.4315796
5: -161.6680145, 125.6641388, -148.5265198, 115.4240494, -277.0920715, 274.1906128
6: -154.9090118, 149.2639313, -142.1847839, 137.1167755, -292.0257874, 291.4487305
7: -168.3108063, 141.9608154, -154.3788910, 130.4327393, -298.7435303, 296.3396606
8: -203.9517212, 139.3747559, -187.2377014, 128.0744019, -332.0261230, 326.6123962
9: -153.7831573, 150.9584045, -141.2319183, 138.5804291, -292.3635864, 292.1903076

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8865759, upper bound: 338.8856684
time: 10.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8839100, upper bound: 338.8837365
time: 11.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -164.9625397, 131.2430420, -167.3192749, 133.1367798, -298.0993042, 298.5623169
1: -138.5303802, 116.4985123, -140.4908600, 118.1707001, -256.7010803, 256.9893494
2: -181.8481903, 117.9277649, -184.4356842, 119.5567169, -301.4048767, 302.3634338
3: -192.9579315, 102.0571747, -195.7241669, 103.5319290, -296.4898682, 297.7813416
4: -177.0283661, 135.3835907, -179.5482178, 137.3009949, -314.3293152, 314.9318237
5: -158.6813965, 123.3680267, -160.9980927, 125.1617432, -283.8431396, 284.3660889
6: -152.0389252, 146.4878845, -154.1716614, 148.5938873, -300.6328125, 300.6595459
7: -165.2204132, 139.3786621, -167.5737152, 141.3495331, -306.5699463, 306.9523621
8: -200.1013794, 136.7763824, -202.8476868, 138.7210388, -338.8223877, 339.6240845
9: -150.9684143, 148.2609253, -153.0993958, 150.3624573, -301.3308411, 301.3602600

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 117

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8893403, upper bound: 338.8885289
time: 12.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8860007, upper bound: 338.8859907
time: 12.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -164.9625397, 131.2430420, -171.8005981, 136.6908875, -301.6534424, 303.0436401
1: -138.5303802, 116.4985123, -144.1874237, 121.2645111, -259.7948914, 260.6859436
2: -181.8481903, 117.9277649, -189.3496704, 122.7352600, -304.5834045, 307.2774048
3: -192.9579315, 102.0571747, -200.9347534, 106.2686844, -299.2265930, 302.9919128
4: -177.0283661, 135.3835907, -184.3222656, 140.9003601, -317.9286804, 319.7058411
5: -158.6813965, 123.3680267, -165.2249146, 128.4273987, -287.1087952, 288.5929565
6: -152.0389252, 146.4878845, -158.2618408, 152.5234833, -304.5624084, 304.7497253
7: -165.2204132, 139.3786621, -172.0182953, 145.0372925, -310.2576904, 311.3969727
8: -200.1013794, 136.7763824, -208.3207703, 142.3851166, -342.4865112, 345.0971680
9: -150.9684143, 148.2609253, -157.1157074, 154.2577667, -305.2261658, 305.3766174

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 117

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8893403, upper bound: 338.8885289
time: 12.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8860007, upper bound: 338.8859907
time: 9.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -169.6725464, 134.9790039, -161.0414886, 128.1629333, -297.8354492, 296.0205078
1: -142.4199829, 119.7520218, -135.2225647, 113.7084045, -256.1283875, 254.9745789
2: -187.0166931, 121.2688217, -177.5196838, 115.1018295, -302.1184998, 298.7884827
3: -198.4376678, 104.9323425, -188.3162079, 99.6560669, -298.0937500, 293.2485352
4: -182.0488739, 139.1704559, -172.7733002, 132.1305237, -314.1793823, 311.9437256
5: -163.1277313, 126.8060074, -154.9340820, 120.4487534, -283.5764771, 281.7400818
6: -156.3396606, 150.6199646, -148.3827972, 143.0303040, -299.3699341, 299.0026855
7: -169.8987885, 143.2599945, -161.2392120, 136.0674286, -305.9662170, 304.4992065
8: -205.8523254, 140.6269684, -195.3334961, 133.5533905, -339.4057007, 335.9604187
9: -155.1974030, 152.3633881, -147.3579865, 144.6859894, -299.8833923, 299.7213135

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8885291, upper bound: 338.8878421
time: 11.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8839100, upper bound: 338.8859950
time: 11.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -169.6725464, 134.9790039, -160.4612579, 127.7120285, -297.3845215, 295.4402466
1: -142.4199829, 119.7520218, -134.6694794, 113.2490692, -255.6690521, 254.4215088
2: -187.0166931, 121.2688217, -176.8370667, 114.6761169, -301.6928101, 298.1058960
3: -198.4376678, 104.9323425, -187.6093292, 99.3285294, -297.7662048, 292.5416870
4: -182.0488739, 139.1704559, -172.1039581, 131.6175995, -313.6664429, 311.2744141
5: -163.1277313, 126.8060074, -154.3497925, 119.9699554, -283.0976868, 281.1557922
6: -156.3396606, 150.6199646, -147.8272552, 142.5088654, -298.8484497, 298.4471130
7: -169.8987885, 143.2599945, -160.6045837, 135.5787659, -305.4775391, 303.8645630
8: -205.8523254, 140.6269684, -194.6675415, 133.0526123, -338.9049377, 335.2944641
9: -155.1974030, 152.3633881, -146.8259125, 144.1055145, -299.3029175, 299.1893005

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8885291, upper bound: 338.8878421
time: 12.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8839100, upper bound: 338.8859950
time: 10.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -162.3107147, 129.1562958, -156.1887360, 124.3368149, -286.6475220, 285.3450317
1: -136.2079468, 114.5396347, -131.0686646, 110.2483368, -246.4562836, 245.6083069
2: -178.8409271, 115.9971466, -172.0791016, 111.5787888, -290.4196777, 288.0762024
3: -189.7822266, 100.4544525, -182.5782776, 96.6923523, -286.4745178, 283.0326538
4: -174.1104889, 133.1433868, -167.5184937, 128.1185150, -302.2290039, 300.6618652
5: -156.1202393, 121.3385239, -150.3061066, 116.8397293, -272.9598999, 271.6446228
6: -149.5482025, 144.1231079, -143.8591003, 138.7284241, -288.2766113, 287.9822083
7: -162.4391022, 137.1290283, -156.2460175, 131.9708405, -294.4099426, 293.3750610
8: -196.8823242, 134.5721436, -189.3331604, 129.5755005, -326.4578247, 323.9052429
9: -148.5132751, 145.7779388, -142.9028473, 140.2792358, -288.7925110, 288.6807861

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8867527, upper bound: 338.8857536
time: 12.94 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8837722, upper bound: 338.8836044
time: 10.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -162.3107147, 129.1562958, -161.1713409, 128.2846832, -290.5953979, 290.3276062
1: -136.2079468, 114.5396347, -135.1784668, 113.6950607, -249.9030151, 249.7181091
2: -178.8409271, 115.9971466, -177.5458984, 115.1165695, -293.9574585, 293.5430298
3: -189.7822266, 100.4544525, -188.3743591, 99.7365341, -289.5187378, 288.8287354
4: -174.1104889, 133.1433868, -172.8305206, 132.1315765, -306.2420654, 305.9739075
5: -156.1202393, 121.3385239, -155.0154877, 120.4851913, -276.6053772, 276.3539734
6: -149.5482025, 144.1231079, -148.4132996, 143.1026917, -292.6508789, 292.5364075
7: -162.4391022, 137.1290283, -161.2020874, 136.0830841, -298.5221863, 298.3310852
8: -196.8823242, 134.5721436, -195.4060211, 133.6465149, -330.5288391, 329.9780884
9: -148.5132751, 145.7779388, -147.3834534, 144.6280365, -293.1412964, 293.1613770

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -338.8867527, upper bound: 338.8857536
time: 12.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -338.8837722, upper bound: 338.8836044
time: 9.94 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.57 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8872254, upper bound: 338.8862086
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8839052, upper bound: 338.8837217
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8872254, upper bound: 338.8862086
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8839052, upper bound: 338.8837217
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8865759, upper bound: 338.8856684
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8839100, upper bound: 338.8837365
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8865759, upper bound: 338.8856684
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8839100, upper bound: 338.8837365
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8893403, upper bound: 338.8885289
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8860007, upper bound: 338.8859907
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8893403, upper bound: 338.8885289
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8860007, upper bound: 338.8859907
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8885291, upper bound: 338.8878421
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8839100, upper bound: 338.8859950
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8885291, upper bound: 338.8878421
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8839100, upper bound: 338.8859950
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8867527, upper bound: 338.8857536
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8837722, upper bound: 338.8836044
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8867527, upper bound: 338.8857536
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.57
Output dim: 7, lower bound: -338.8837722, upper bound: 338.8836044
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.57
Output dim: 7, lower bound: -338.9035845, upper bound: 338.9035845
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.57
Output dim: 7, lower bound: -338.9035845, upper bound: 338.9035845
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.57
Output dim: 7, lower bound: -338.8874235, upper bound: 338.8881357
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=341.21295166015625
rel_dist={7: [-338.9336846765279, 338.9336846761764]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 2309.69 seconds
