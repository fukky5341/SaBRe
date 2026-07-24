## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 154.56034074419998
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957)
1: (-70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644)
2: (-94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537)
3: (-99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775)
4: (-103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431)
5: (-81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278)
6: (-83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274)
7: (-88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160)
8: (-104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331)
9: (-84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249)

## BASE Result
execution time: IAR + LP analysis = 1.49 + 9.89 = 11.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -154.7151429, upper bound: 154.7151428


# Binary Search by BASE starts (time budget: 1988.62 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=168.67294311523438
rel_dist={4: [-154.71512507779116, 154.715125091485]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=168.67294311523438
rel_dist={4: [-154.7150558205383, 154.7150558205383]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=168.67294311523438
rel_dist={4: [-154.71496529634885, 154.71496529634885]}

## Binary Search Result
Binary search time: 45.65 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1942.97 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7098576, upper bound: 154.7108330
time: 9.37 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7136990, upper bound: 154.7136990
time: 8.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.91 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.91
Output dim: 4, lower bound: -154.7098576, upper bound: 154.7108330
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.91
Output dim: 4, lower bound: -154.7136990, upper bound: 154.7136990

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -78.1110840, 60.8416214, -87.2482681, 68.0545349, -146.1656189, 148.0898895
1: -62.1475143, 55.3199234, -69.8258057, 61.7979546, -123.9454651, 125.1457291
2: -83.7355194, 57.4565086, -93.7172241, 63.9473305, -147.6828461, 151.1737366
3: -88.4660645, 49.0245056, -99.0214081, 54.7005157, -143.1665802, 148.0459137
4: -92.5861588, 57.3921318, -102.5926132, 64.8089981, -157.3951111, 159.9847412
5: -71.8089142, 58.6708832, -80.4441910, 65.4860687, -137.2949524, 139.1150665
6: -73.9255219, 68.6816177, -82.4278488, 76.7657242, -150.6912537, 151.1094208
7: -78.4739304, 66.8039246, -87.6206512, 74.4884872, -152.9624176, 154.4245605
8: -92.7796097, 64.0383682, -103.8875198, 71.7666779, -164.5462646, 167.9258575
9: -75.2036514, 66.6150513, -83.6389999, 74.8037491, -150.0074005, 150.2540283

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6595033, upper bound: 154.6487648
time: 12.26 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7084860, upper bound: 154.7094073
time: 8.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -82.4793930, 64.2899170, -87.6090240, 68.3400421, -150.8194275, 151.8989410
1: -65.8094025, 58.4177246, -70.1306915, 62.0538063, -127.8632050, 128.5484161
2: -88.5034561, 60.5652313, -94.1120682, 64.2036057, -152.7070618, 154.6773071
3: -93.5097885, 51.7365265, -99.4386597, 54.9249878, -148.4347687, 151.1751862
4: -97.4018021, 60.9177780, -102.9858475, 65.1039047, -162.5056763, 163.9036255
5: -75.9326935, 61.9275970, -80.7860260, 65.7552795, -141.6879730, 142.7136230
6: -77.9988632, 72.5447617, -82.7633438, 77.0854797, -155.0843048, 155.3080750
7: -82.8456421, 70.4815216, -87.9819107, 74.7917480, -157.6373749, 158.4634399
8: -98.0842896, 67.7380829, -104.3273849, 72.0727997, -170.1570892, 172.0654144
9: -79.2458572, 70.5169907, -83.9713287, 75.1288223, -154.3746796, 154.4882812

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7108330, upper bound: 154.7098576
time: 11.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7108330, upper bound: 154.7136990
time: 12.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.79 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.79
Output dim: 4, lower bound: -154.6595033, upper bound: 154.6487648
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.79
Output dim: 4, lower bound: -154.7084860, upper bound: 154.7094073
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.79
Output dim: 4, lower bound: -154.7108330, upper bound: 154.7098576
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.79
Output dim: 4, lower bound: -154.7108330, upper bound: 154.7136990

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -74.9655609, 58.3245316, -69.2697983, 53.5024834, -128.4680328, 127.5943298
1: -59.4863510, 53.0892296, -54.3270149, 49.0449066, -108.5312576, 107.4162445
2: -80.2550964, 55.2302017, -73.6418076, 51.3492546, -131.6043549, 128.8719940
3: -84.7990265, 47.0973969, -77.8227386, 43.7646904, -128.5637207, 124.9201355
4: -89.2604828, 54.7187767, -84.4518738, 48.7029800, -137.9634399, 139.1706390
5: -68.7833405, 56.3032494, -62.8120041, 51.9065247, -120.6898651, 119.1152496
6: -71.0413895, 65.8976669, -66.0975189, 60.8134079, -131.8547821, 131.9951782
7: -75.3352585, 64.1638794, -69.6962814, 59.4613647, -134.7965851, 133.8601532
8: -88.9642181, 61.3894119, -81.9105225, 56.4938202, -145.4580231, 143.2999268
9: -72.3438110, 63.7382584, -67.6037292, 57.9381104, -130.2819214, 131.3419647

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6315696, upper bound: 154.6320730
time: 9.39 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6315696, upper bound: 154.6487648
time: 9.61 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -78.1110840, 60.8416214, -84.7749023, 66.0824585, -144.1935425, 145.6165161
1: -62.1475143, 55.3199234, -67.7372513, 60.0460396, -122.1935577, 123.0571747
2: -83.7355194, 57.4565086, -90.9870987, 62.1933289, -145.9288330, 148.4435883
3: -88.4660645, 49.0245056, -96.1408920, 53.1828156, -141.6488800, 145.1654053
4: -92.5861588, 57.3921318, -99.9591293, 62.7354660, -155.3216248, 157.3512573
5: -71.8089142, 58.6708832, -78.0716400, 63.6331635, -135.4420776, 136.7425232
6: -73.9255219, 68.6816177, -80.1588593, 74.5752258, -148.5007324, 148.8404541
7: -78.4739304, 66.8039246, -85.1487274, 72.4089508, -150.8828583, 151.9526062
8: -92.7796097, 64.0383682, -100.8948288, 69.6990128, -162.4786072, 164.9331665
9: -75.2036514, 66.6150513, -81.3817062, 72.5503616, -147.7540131, 147.9967499

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6486314, upper bound: 154.6604599
time: 14.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6486314, upper bound: 154.7094073
time: 13.62 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -82.4793930, 64.2899170, -78.1110840, 60.8416214, -143.3210144, 142.4010010
1: -65.8094025, 58.4177246, -62.1475143, 55.3199234, -121.1293259, 120.5652390
2: -88.5034561, 60.5652313, -83.7355194, 57.4565086, -145.9599457, 144.3007355
3: -93.5097885, 51.7365265, -88.4660645, 49.0245056, -142.5343018, 140.2025757
4: -97.4018021, 60.9177780, -92.5861588, 57.3921318, -154.7938995, 153.5039368
5: -75.9326935, 61.9275970, -71.8089142, 58.6708832, -134.6035767, 133.7365112
6: -77.9988632, 72.5447617, -73.9255219, 68.6816177, -146.6804352, 146.4702606
7: -82.8456421, 70.4815216, -78.4739304, 66.8039246, -149.6495209, 148.9554443
8: -98.0842896, 67.7380829, -92.7796097, 64.0383682, -162.1226501, 160.5176544
9: -79.2458572, 70.5169907, -75.2036514, 66.6150513, -145.8609009, 145.7206421

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6487648, upper bound: 154.6595033
time: 16.87 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7094073, upper bound: 154.7084860
time: 10.62 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -82.4793930, 64.2899170, -82.4793930, 64.2899170, -146.7693176, 146.7693176
1: -65.8094025, 58.4177246, -65.8094025, 58.4177246, -124.2271194, 124.2271194
2: -88.5034561, 60.5652313, -88.5034561, 60.5652313, -149.0686951, 149.0686951
3: -93.5097885, 51.7365265, -93.5097885, 51.7365265, -145.2463074, 145.2463074
4: -97.4018021, 60.9177780, -97.4018021, 60.9177780, -158.3195496, 158.3195496
5: -75.9326935, 61.9275970, -75.9326935, 61.9275970, -137.8602905, 137.8602905
6: -77.9988632, 72.5447617, -77.9988632, 72.5447617, -150.5435638, 150.5435638
7: -82.8456421, 70.4815216, -82.8456421, 70.4815216, -153.3271179, 153.3271179
8: -98.0842896, 67.7380829, -98.0842896, 67.7380829, -165.8223419, 165.8223419
9: -79.2458572, 70.5169907, -79.2458572, 70.5169907, -149.7628326, 149.7628174

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6487648, upper bound: 154.6622802
time: 11.44 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7094073, upper bound: 154.7125269
time: 11.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 4, lower bound: -154.6315696, upper bound: 154.6320730
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 4, lower bound: -154.6315696, upper bound: 154.6487648
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 4, lower bound: -154.6486314, upper bound: 154.6604599
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 4, lower bound: -154.6486314, upper bound: 154.7094073
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 4, lower bound: -154.6487648, upper bound: 154.6595033
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 4, lower bound: -154.7094073, upper bound: 154.7084860
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 4, lower bound: -154.6487648, upper bound: 154.6622802
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 4, lower bound: -154.7094073, upper bound: 154.7125269

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -61.0134735, 46.9926720, -69.2697983, 53.5024834, -114.5159607, 116.2624588
1: -47.4360580, 43.2015343, -54.3270149, 49.0449066, -96.4809570, 97.5285492
2: -64.6352997, 45.4873924, -73.6418076, 51.3492546, -115.9845581, 119.1291962
3: -68.2957535, 38.6569023, -77.8227386, 43.7646904, -112.0604401, 116.4796448
4: -75.4870224, 41.9992981, -84.4518738, 48.7029800, -124.1899719, 126.4511719
5: -55.0194321, 45.7228470, -62.8120041, 51.9065247, -106.9259491, 108.5348511
6: -58.4250526, 53.5161018, -66.0975189, 60.8134079, -119.2384567, 119.6136169
7: -61.4343147, 52.4935989, -69.6962814, 59.4613647, -120.8956757, 122.1898804
8: -71.8887329, 49.5572662, -81.9105225, 56.4938202, -128.3825531, 131.4677734
9: -59.9705887, 50.5815125, -67.6037292, 57.9381104, -117.9086914, 118.1852417

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6310083
time: 8.51 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6320730
time: 7.51 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -75.6670303, 58.8925705, -69.2697983, 53.5024834, -129.1695099, 128.1623688
1: -60.0846748, 53.5883102, -54.3270149, 49.0449066, -109.1295776, 107.9153290
2: -81.0385742, 55.7261200, -73.6418076, 51.3492546, -132.3878326, 129.3679199
3: -85.6200790, 47.5263901, -77.8227386, 43.7646904, -129.3847656, 125.3491287
4: -89.9919891, 55.3355865, -84.4518738, 48.7029800, -138.6949615, 139.7874603
5: -69.4646378, 56.8362579, -62.8120041, 51.9065247, -121.3711624, 119.6482620
6: -71.6836777, 66.5193253, -66.0975189, 60.8134079, -132.4970856, 132.6168518
7: -76.0339203, 64.7512207, -69.6962814, 59.4613647, -135.4952698, 134.4474945
8: -89.8267441, 61.9960556, -81.9105225, 56.4938202, -146.3205566, 143.9065857
9: -72.9766541, 64.3889694, -67.6037292, 57.9381104, -130.9147644, 131.9927063

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6475937
time: 9.69 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6487648
time: 10.37 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -61.0134735, 46.9926720, -84.7749023, 66.0824585, -127.0959320, 131.7675781
1: -47.4360580, 43.2015343, -67.7372513, 60.0460396, -107.4820862, 110.9387741
2: -64.6352997, 45.4873924, -90.9870987, 62.1933289, -126.8285904, 136.4744873
3: -68.2957535, 38.6569023, -96.1408920, 53.1828156, -121.4785614, 134.7977905
4: -75.4870224, 41.9992981, -99.9591293, 62.7354660, -138.2224884, 141.9584045
5: -55.0194321, 45.7228470, -78.0716400, 63.6331635, -118.6525955, 123.7944870
6: -58.4250526, 53.5161018, -80.1588593, 74.5752258, -133.0002747, 133.6749573
7: -61.4343147, 52.4935989, -85.1487274, 72.4089508, -133.8432617, 137.6423187
8: -71.8887329, 49.5572662, -100.8948288, 69.6990128, -141.5877380, 150.4520569
9: -59.9705887, 50.5815125, -81.3817062, 72.5503616, -132.5209503, 131.9632111

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6588348
time: 9.22 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6604599
time: 10.28 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -75.6670303, 58.8925705, -84.7749023, 66.0824585, -141.7494659, 143.6674805
1: -60.0846748, 53.5883102, -67.7372513, 60.0460396, -120.1307068, 121.3255615
2: -81.0385742, 55.7261200, -90.9870987, 62.1933289, -143.2319031, 146.7132263
3: -85.6200790, 47.5263901, -96.1408920, 53.1828156, -138.8028870, 143.6672821
4: -89.9919891, 55.3355865, -99.9591293, 62.7354660, -152.7274475, 155.2947083
5: -69.4646378, 56.8362579, -78.0716400, 63.6331635, -133.0978088, 134.9078979
6: -71.6836777, 66.5193253, -80.1588593, 74.5752258, -146.2588959, 146.6781921
7: -76.0339203, 64.7512207, -85.1487274, 72.4089508, -148.4428558, 149.8999176
8: -89.8267441, 61.9960556, -100.8948288, 69.6990128, -159.5257568, 162.8908691
9: -72.9766541, 64.3889694, -81.3817062, 72.5503616, -145.5270081, 145.7706757

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6310083, upper bound: 154.7071330
time: 10.80 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6310083, upper bound: 154.7093970
time: 9.62 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -64.8345795, 50.0017662, -74.9655609, 58.3245316, -123.1591110, 124.9673080
1: -50.6013718, 45.8979492, -59.4863510, 53.0892296, -103.6905975, 105.3843002
2: -68.7912064, 48.2071762, -80.2550964, 55.2302017, -124.0213776, 128.4622803
3: -72.6958542, 41.0185890, -84.7990265, 47.0973969, -119.7932510, 125.8176117
4: -79.6702118, 45.0614700, -89.2604828, 54.7187767, -134.3889923, 134.3219299
5: -58.6162605, 48.5807838, -68.7833405, 56.3032494, -114.9194946, 117.3641205
6: -61.9852524, 56.8900986, -71.0413895, 65.8976669, -127.8828964, 127.9314880
7: -65.2527542, 55.7268562, -75.3352585, 64.1638794, -129.4166260, 131.0620880
8: -76.5235519, 52.7604256, -88.9642181, 61.3894119, -137.9129181, 141.7246399
9: -63.5092888, 53.9645233, -72.3438110, 63.7382584, -127.2475433, 126.3083344

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6320730, upper bound: 154.6315696
time: 7.72 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6320730, upper bound: 154.6595033
time: 8.13 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -80.0182724, 62.3286095, -78.1110840, 60.8416214, -140.8598938, 140.4396973
1: -63.7334671, 56.6745224, -62.1475143, 55.3199234, -119.0533905, 118.8220367
2: -85.7877960, 58.8206749, -83.7355194, 57.4565086, -143.2442780, 142.5561371
3: -90.6423111, 50.2274284, -88.4660645, 49.0245056, -139.6668091, 138.6934814
4: -94.7824707, 58.8527756, -92.5861588, 57.3921318, -152.1746063, 151.4389191
5: -73.5729752, 60.0822754, -71.8089142, 58.6708832, -132.2438660, 131.8911896
6: -75.7399292, 70.3668365, -73.9255219, 68.6816177, -144.4215240, 144.2923584
7: -80.3863068, 68.4134293, -78.4739304, 66.8039246, -147.1901855, 146.8873596
8: -95.1086044, 65.6820602, -92.7796097, 64.0383682, -159.1469727, 158.4616547
9: -77.0006256, 68.2752304, -75.2036514, 66.6150513, -143.6156616, 143.4788818

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6604599, upper bound: 154.6486314
time: 15.14 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6604599, upper bound: 154.7084860
time: 19.36 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -64.8345795, 50.0017662, -79.3222351, 61.7629852, -126.5975647, 129.3240051
1: -50.6013718, 45.8979492, -63.1374741, 56.1772346, -106.7785950, 109.0354233
2: -68.7912064, 48.2071762, -85.0081100, 58.3289642, -127.1201553, 133.2152863
3: -72.6958542, 41.0185890, -89.8246765, 49.8009148, -122.4967651, 130.8432465
4: -79.6702118, 45.0614700, -94.0590820, 58.2359390, -137.9061584, 139.1205292
5: -58.6162605, 48.5807838, -72.8948212, 59.5519371, -118.1681976, 121.4756012
6: -61.9852524, 56.8900986, -75.1028290, 69.7486191, -131.7338562, 131.9929199
7: -65.2527542, 55.7268562, -79.6921616, 67.8294754, -133.0822144, 135.4190063
8: -76.5235519, 52.7604256, -94.2501907, 65.0781326, -141.6016846, 147.0106201
9: -63.5092888, 53.9645233, -76.3724060, 67.6276016, -131.1368866, 130.3369293

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6338237, upper bound: 154.6338265
time: 7.63 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6338237, upper bound: 154.6622802
time: 10.05 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -80.0182724, 62.3286095, -82.4793930, 64.2899170, -144.3081818, 144.8079987
1: -63.7334671, 56.6745224, -65.8094025, 58.4177246, -122.1511917, 122.4839249
2: -85.7877960, 58.8206749, -88.5034561, 60.5652313, -146.3530273, 147.3241119
3: -90.6423111, 50.2274284, -93.5097885, 51.7365265, -142.3788452, 143.7372131
4: -94.7824707, 58.8527756, -97.4018021, 60.9177780, -155.7002563, 156.2545624
5: -73.5729752, 60.0822754, -75.9326935, 61.9275970, -135.5005798, 136.0149689
6: -75.7399292, 70.3668365, -77.9988632, 72.5447617, -148.2846680, 148.3656769
7: -80.3863068, 68.4134293, -82.8456421, 70.4815216, -150.8677979, 151.2590637
8: -95.1086044, 65.6820602, -98.0842896, 67.7380829, -162.8466797, 163.7663269
9: -77.0006256, 68.2752304, -79.2458572, 70.5169907, -147.5175629, 147.5210876

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6626034, upper bound: 154.6513381
time: 13.76 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6626034, upper bound: 154.7125269
time: 13.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 28.66 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6310083
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6320730
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6475937
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6487648
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6588348
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6310083, upper bound: 154.6604599
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6310083, upper bound: 154.7071330
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6310083, upper bound: 154.7093970
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6320730, upper bound: 154.6315696
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6320730, upper bound: 154.6595033
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6604599, upper bound: 154.6486314
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6604599, upper bound: 154.7084860
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6338237, upper bound: 154.6338265
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6338237, upper bound: 154.6622802
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6626034, upper bound: 154.6513381
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 4, lower bound: -154.6626034, upper bound: 154.7125269

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -61.0134735, 46.9926720, -61.0134735, 46.9926720, -108.0061340, 108.0061340
1: -47.4360580, 43.2015343, -47.4360580, 43.2015343, -90.6375809, 90.6375809
2: -64.6352997, 45.4873924, -64.6352997, 45.4873924, -110.1226807, 110.1226807
3: -68.2957535, 38.6569023, -68.2957535, 38.6569023, -106.9526520, 106.9526520
4: -75.4870224, 41.9992981, -75.4870224, 41.9992981, -117.4863129, 117.4863205
5: -55.0194321, 45.7228470, -55.0194321, 45.7228470, -100.7422714, 100.7422714
6: -58.4250526, 53.5161018, -58.4250526, 53.5161018, -111.9411545, 111.9411545
7: -61.4343147, 52.4935989, -61.4343147, 52.4935989, -113.9279175, 113.9279175
8: -71.8887329, 49.5572662, -71.8887329, 49.5572662, -121.4459915, 121.4459991
9: -59.9705887, 50.5815125, -59.9705887, 50.5815125, -110.5520935, 110.5520935

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6297810, upper bound: 154.6295296
time: 9.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6290007, upper bound: 154.6290007
time: 7.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -61.0134735, 46.9926720, -64.8345795, 50.0017662, -111.0152435, 111.8272247
1: -47.4360580, 43.2015343, -50.6013718, 45.8979492, -93.3339996, 93.8028946
2: -64.6352997, 45.4873924, -68.7912064, 48.2071762, -112.8424759, 114.2785873
3: -68.2957535, 38.6569023, -72.6958542, 41.0185890, -109.3143311, 111.3527527
4: -75.4870224, 41.9992981, -79.6702118, 45.0614700, -120.5484772, 121.6695099
5: -55.0194321, 45.7228470, -58.6162605, 48.5807838, -103.6002197, 104.3390961
6: -58.4250526, 53.5161018, -61.9852524, 56.8900986, -115.3151550, 115.5013351
7: -61.4343147, 52.4935989, -65.2527542, 55.7268562, -117.1611633, 117.7463531
8: -71.8887329, 49.5572662, -76.5235519, 52.7604256, -124.6491547, 126.0807953
9: -59.9705887, 50.5815125, -63.5092888, 53.9645233, -113.9351120, 114.0908051

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6297810, upper bound: 154.6307139
time: 10.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6290007, upper bound: 154.6303051
time: 6.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -75.6670303, 58.8925705, -61.0134735, 46.9926720, -122.6596985, 119.9060440
1: -60.0846748, 53.5883102, -47.4360580, 43.2015343, -103.2862015, 101.0243683
2: -81.0385742, 55.7261200, -64.6352997, 45.4873924, -126.5259552, 120.3614197
3: -85.6200790, 47.5263901, -68.2957535, 38.6569023, -124.2769775, 115.8221436
4: -89.9919891, 55.3355865, -75.4870224, 41.9992981, -131.9912872, 130.8226013
5: -69.4646378, 56.8362579, -55.0194321, 45.7228470, -115.1874847, 111.8556824
6: -71.6836777, 66.5193253, -58.4250526, 53.5161018, -125.1997833, 124.9443817
7: -76.0339203, 64.7512207, -61.4343147, 52.4935989, -128.5275269, 126.1855316
8: -89.8267441, 61.9960556, -71.8887329, 49.5572662, -139.3839874, 133.8847961
9: -72.9766541, 64.3889694, -59.9705887, 50.5815125, -123.5581665, 124.3595581

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6491972, upper bound: 154.6371642
time: 11.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6509345, upper bound: 154.6385096
time: 11.26 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -75.6670303, 58.8925705, -64.8345795, 50.0017662, -125.6687927, 123.7271500
1: -60.0846748, 53.5883102, -50.6013718, 45.8979492, -105.9826202, 104.1896820
2: -81.0385742, 55.7261200, -68.7912064, 48.2071762, -129.2457581, 124.5173264
3: -85.6200790, 47.5263901, -72.6958542, 41.0185890, -126.6386719, 120.2222443
4: -89.9919891, 55.3355865, -79.6702118, 45.0614700, -135.0534515, 135.0057983
5: -69.4646378, 56.8362579, -58.6162605, 48.5807838, -118.0454254, 115.4525070
6: -71.6836777, 66.5193253, -61.9852524, 56.8900986, -128.5737762, 128.5045319
7: -76.0339203, 64.7512207, -65.2527542, 55.7268562, -131.7607727, 130.0039520
8: -89.8267441, 61.9960556, -76.5235519, 52.7604256, -142.5871735, 138.5196075
9: -72.9766541, 64.3889694, -63.5092888, 53.9645233, -126.9411697, 127.8982544

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6491972, upper bound: 154.6377568
time: 16.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6509345, upper bound: 154.6393697
time: 13.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -61.0134735, 46.9926720, -75.6542282, 58.8826332, -119.8961029, 122.6468811
1: -47.4360580, 43.2015343, -60.0750198, 53.5793571, -101.0154114, 103.2765427
2: -64.6352997, 45.4873924, -81.0251083, 55.7159729, -120.3512573, 126.5124969
3: -68.2957535, 38.6569023, -85.6065979, 47.5181313, -115.8138885, 124.2635040
4: -75.4870224, 41.9992981, -89.9739685, 55.3282089, -130.8152008, 131.9732666
5: -55.0194321, 45.7228470, -69.4538879, 56.8267860, -111.8462143, 115.1767349
6: -58.4250526, 53.5161018, -71.6709976, 66.5078812, -124.9329300, 125.1871033
7: -61.4343147, 52.4935989, -76.0209503, 64.7398987, -126.1742096, 128.5145569
8: -71.8887329, 49.5572662, -89.8109131, 61.9851761, -133.8739014, 139.3681641
9: -59.9705887, 50.5815125, -72.9631805, 64.3790359, -124.3496246, 123.5446930

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6456139, upper bound: 154.6563278
time: 13.11 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6451958, upper bound: 154.6562335
time: 10.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -61.0134735, 46.9926720, -80.0182724, 62.3286095, -123.3420868, 127.0109329
1: -47.4360580, 43.2015343, -63.7334671, 56.6745224, -104.1105804, 106.9349899
2: -64.6352997, 45.4873924, -85.7877960, 58.8206749, -123.4559555, 131.2751770
3: -68.2957535, 38.6569023, -90.6423111, 50.2274284, -118.5231781, 129.2992096
4: -75.4870224, 41.9992981, -94.7824707, 58.8527756, -134.3397827, 136.7817688
5: -55.0194321, 45.7228470, -73.5729752, 60.0822754, -115.1017075, 119.2958221
6: -58.4250526, 53.5161018, -75.7399292, 70.3668365, -128.7918854, 129.2560120
7: -61.4343147, 52.4935989, -80.3863068, 68.4134293, -129.8477478, 132.8798981
8: -71.8887329, 49.5572662, -95.1086044, 65.6820602, -137.5708008, 144.6658630
9: -59.9705887, 50.5815125, -77.0006256, 68.2752304, -128.2458191, 127.5821228

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6456139, upper bound: 154.6563278
time: 13.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6451958, upper bound: 154.6573665
time: 11.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -75.6670303, 58.8925705, -75.6542282, 58.8826332, -134.5496521, 134.5467682
1: -60.0846748, 53.5883102, -60.0750198, 53.5793571, -113.6640320, 113.6633301
2: -81.0385742, 55.7261200, -81.0251083, 55.7159729, -136.7545166, 136.7512207
3: -85.6200790, 47.5263901, -85.6065979, 47.5181313, -133.1382141, 133.1329956
4: -89.9919891, 55.3355865, -89.9739685, 55.3282089, -145.3201752, 145.3095551
5: -69.4646378, 56.8362579, -69.4538879, 56.8267860, -126.2914200, 126.2901459
6: -71.6836777, 66.5193253, -71.6709976, 66.5078812, -138.1915588, 138.1903229
7: -76.0339203, 64.7512207, -76.0209503, 64.7398987, -140.7738190, 140.7721558
8: -89.8267441, 61.9960556, -89.8109131, 61.9851761, -151.8118896, 151.8069611
9: -72.9766541, 64.3889694, -72.9631805, 64.3790359, -137.3556824, 137.3521423

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7017676, upper bound: 154.7021835
time: 9.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7045251, upper bound: 154.7045250
time: 9.30 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -75.6670303, 58.8925705, -80.0182724, 62.3286095, -137.9956360, 138.9108429
1: -60.0846748, 53.5883102, -63.7334671, 56.6745224, -116.7592010, 117.3217773
2: -81.0385742, 55.7261200, -85.7877960, 58.8206749, -139.8592529, 141.5139008
3: -85.6200790, 47.5263901, -90.6423111, 50.2274284, -135.8475037, 138.1687012
4: -89.9919891, 55.3355865, -94.7824707, 58.8527756, -148.8447571, 150.1180573
5: -69.4646378, 56.8362579, -73.5729752, 60.0822754, -129.5469055, 130.4092407
6: -71.6836777, 66.5193253, -75.7399292, 70.3668365, -142.0505066, 142.2592468
7: -76.0339203, 64.7512207, -80.3863068, 68.4134293, -144.4473572, 145.1374817
8: -89.8267441, 61.9960556, -95.1086044, 65.6820602, -155.5087891, 157.1046600
9: -72.9766541, 64.3889694, -77.0006256, 68.2752304, -141.2518768, 141.3895874

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7017676, upper bound: 154.7041571
time: 9.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7045251, upper bound: 154.7064662
time: 8.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -64.8345795, 50.0017662, -61.0134735, 46.9926720, -111.8272400, 111.0152359
1: -50.6013718, 45.8979492, -47.4360580, 43.2015343, -93.8028946, 93.3339996
2: -68.7912064, 48.2071762, -64.6352997, 45.4873924, -114.2785873, 112.8424759
3: -72.6958542, 41.0185890, -68.2957535, 38.6569023, -111.3527527, 109.3143311
4: -79.6702118, 45.0614700, -75.4870224, 41.9992981, -121.6695099, 120.5484772
5: -58.6162605, 48.5807838, -55.0194321, 45.7228470, -104.3390961, 103.6002197
6: -61.9852524, 56.8900986, -58.4250526, 53.5161018, -115.5013428, 115.3151550
7: -65.2527542, 55.7268562, -61.4343147, 52.4935989, -117.7463531, 117.1611710
8: -76.5235519, 52.7604256, -71.8887329, 49.5572662, -126.0807724, 124.6491547
9: -63.5092888, 53.9645233, -59.9705887, 50.5815125, -114.0907974, 113.9351120

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6241401, upper bound: 154.6251451
time: 8.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6289172, upper bound: 154.6283294
time: 7.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -64.8345795, 50.0017662, -75.6670303, 58.8925705, -123.7271500, 125.6687927
1: -50.6013718, 45.8979492, -60.0846748, 53.5883102, -104.1896820, 105.9826202
2: -68.7912064, 48.2071762, -81.0385742, 55.7261200, -124.5173264, 129.2457581
3: -72.6958542, 41.0185890, -85.6200790, 47.5263901, -120.2222443, 126.6386642
4: -79.6702118, 45.0614700, -89.9919891, 55.3355865, -135.0057983, 135.0534515
5: -58.6162605, 48.5807838, -69.4646378, 56.8362579, -115.4525146, 118.0454254
6: -61.9852524, 56.8900986, -71.6836777, 66.5193253, -128.5045471, 128.5737762
7: -65.2527542, 55.7268562, -76.0339203, 64.7512207, -130.0039520, 131.7607727
8: -76.5235519, 52.7604256, -89.8267441, 61.9960556, -138.5196075, 142.5871735
9: -63.5092888, 53.9645233, -72.9766541, 64.3889694, -127.8982544, 126.9411774

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6241401, upper bound: 154.6532931
time: 9.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6289172, upper bound: 154.6565475
time: 6.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -80.0182724, 62.3286095, -61.0134735, 46.9926720, -127.0109406, 123.3420868
1: -63.7334671, 56.6745224, -47.4360580, 43.2015343, -106.9349899, 104.1105804
2: -85.7877960, 58.8206749, -64.6352997, 45.4873924, -131.2751770, 123.4559555
3: -90.6423111, 50.2274284, -68.2957535, 38.6569023, -129.2992096, 118.5231781
4: -94.7824707, 58.8527756, -75.4870224, 41.9992981, -136.7817688, 134.3397980
5: -73.5729752, 60.0822754, -55.0194321, 45.7228470, -119.2958221, 115.1017075
6: -75.7399292, 70.3668365, -58.4250526, 53.5161018, -129.2560272, 128.7918854
7: -80.3863068, 68.4134293, -61.4343147, 52.4935989, -132.8798828, 129.8477478
8: -95.1086044, 65.6820602, -71.8887329, 49.5572662, -144.6658478, 137.5708008
9: -77.0006256, 68.2752304, -59.9705887, 50.5815125, -127.5821381, 128.2458191

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6511263, upper bound: 154.6387726
time: 13.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6521804, upper bound: 154.6394903
time: 16.53 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 31.54 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6297810, upper bound: 154.6295296
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6290007, upper bound: 154.6290007
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6297810, upper bound: 154.6307139
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6290007, upper bound: 154.6303051
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6491972, upper bound: 154.6371642
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6509345, upper bound: 154.6385096
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6491972, upper bound: 154.6377568
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6509345, upper bound: 154.6393697
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6456139, upper bound: 154.6563278
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6451958, upper bound: 154.6562335
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6456139, upper bound: 154.6563278
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6451958, upper bound: 154.6573665
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.7017676, upper bound: 154.7021835
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.7045251, upper bound: 154.7045250
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.7017676, upper bound: 154.7041571
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.7045251, upper bound: 154.7064662
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6241401, upper bound: 154.6251451
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6289172, upper bound: 154.6283294
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6241401, upper bound: 154.6532931
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6289172, upper bound: 154.6565475
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6511263, upper bound: 154.6387726
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 4, lower bound: -154.6521804, upper bound: 154.6394903
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.54
Output dim: 4, lower bound: -154.6604599, upper bound: 154.7084860
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.54
Output dim: 4, lower bound: -154.6338237, upper bound: 154.6338265
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.54
Output dim: 4, lower bound: -154.6338237, upper bound: 154.6622802
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.54
Output dim: 4, lower bound: -154.6626034, upper bound: 154.6513381
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.54
Output dim: 4, lower bound: -154.6626034, upper bound: 154.7125269
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=168.67294311523438
rel_dist={4: [-154.71512507779116, 154.715125091485]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7094079, upper bound: 154.7100752
time: 11.23 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7136831, upper bound: 154.7136831
time: 10.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 22.23 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 22.23
Output dim: 4, lower bound: -154.7094079, upper bound: 154.7100752
IS_A2, status: Status.UNKNOWN, split count: 1, time: 22.23
Output dim: 4, lower bound: -154.7136831, upper bound: 154.7136831

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -78.1110840, 60.8416214, -84.2285233, 65.6670074, -143.7780914, 145.0701447
1: -62.1475143, 55.3199234, -67.2769165, 59.6561127, -121.8036270, 122.5968399
2: -83.7355194, 57.4565086, -90.4149170, 61.8032837, -145.5387573, 147.8713989
3: -88.4660645, 49.0245056, -95.5290833, 52.8229332, -141.2890015, 144.5535889
4: -92.5861588, 57.3921318, -99.3052063, 62.3420029, -154.9281464, 156.6973267
5: -71.8089142, 58.6708832, -77.5844879, 63.2328873, -135.0417938, 136.2553711
6: -73.9255219, 68.6816177, -79.6215744, 74.0920715, -148.0175781, 148.3031616
7: -78.4739304, 66.8039246, -84.5962753, 71.9510803, -150.4250031, 151.4001923
8: -92.7796097, 64.0383682, -100.2095566, 69.2089310, -161.9885254, 164.2479248
9: -75.2036514, 66.6150513, -80.8588943, 72.0851746, -147.2888184, 147.4739380

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6469897, upper bound: 154.6407124
time: 12.58 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7079469, upper bound: 154.7086554
time: 10.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -82.4793930, 64.2899170, -85.7685165, 66.8867722, -149.3661652, 150.0584259
1: -65.8094025, 58.4177246, -68.5805511, 60.7502022, -126.5595627, 126.9982758
2: -88.5034561, 60.5652313, -92.1007690, 62.8974571, -151.4009094, 152.6660004
3: -93.5097885, 51.7365265, -97.3124008, 53.7803459, -147.2901306, 149.0489197
4: -97.4018021, 60.9177780, -100.9806824, 63.6041298, -161.0059357, 161.8984528
5: -75.9326935, 61.9275970, -79.0450439, 64.3821335, -140.3148193, 140.9726410
6: -77.9988632, 72.5447617, -81.0534210, 75.4568710, -153.4556885, 153.5981293
7: -82.8456421, 70.4815216, -86.1400604, 73.2447205, -156.0903320, 156.6215668
8: -98.0842896, 67.7380829, -102.0872650, 70.5174026, -168.6016693, 169.8253326
9: -79.2458572, 70.5169907, -82.2753906, 73.4750595, -152.7209167, 152.7923279

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6498743, upper bound: 154.6433765
time: 12.79 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7124971, upper bound: 154.7124971
time: 12.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.78 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 4, lower bound: -154.6469897, upper bound: 154.6407124
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 4, lower bound: -154.7079469, upper bound: 154.7086554
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 4, lower bound: -154.6498743, upper bound: 154.6433765
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 4, lower bound: -154.7124971, upper bound: 154.7124971

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -70.4228745, 54.6894150, -66.4248047, 51.2497292, -121.6725922, 121.1142044
1: -55.6461411, 49.8676529, -51.9296989, 47.0211754, -102.6673050, 101.7973251
2: -75.2288055, 52.0178146, -70.5255814, 49.3287354, -124.5575256, 122.5433807
3: -79.5048599, 44.3138008, -74.5316162, 42.0020638, -121.5069199, 118.8454132
4: -84.4666595, 50.8490295, -81.3808289, 46.3640289, -130.8306732, 132.2298279
5: -64.4108276, 52.8829575, -60.1173477, 49.7720261, -114.1828537, 113.0003052
6: -66.8773804, 61.8791733, -63.4553528, 58.2953033, -125.1726761, 125.3345108
7: -70.8017960, 60.3550034, -66.8436661, 57.0650177, -127.8668137, 127.1986694
8: -83.4577637, 57.5659256, -78.4479218, 54.0891304, -137.5468903, 136.0138397
9: -68.2179413, 59.5853157, -64.9767456, 55.3823090, -123.6002426, 124.5620422

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6359814, upper bound: 154.6294650
time: 12.76 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6370444, upper bound: 154.6303505
time: 13.32 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -78.1110840, 60.8416214, -81.7621460, 63.7012024, -141.8122864, 142.6037598
1: -62.1475143, 55.3199234, -65.1951294, 57.9080200, -120.0555344, 120.5150528
2: -83.7355194, 57.4565086, -87.6928787, 60.0546951, -143.7901917, 145.1493683
3: -88.4660645, 49.0245056, -92.6549377, 51.3104820, -139.7765503, 141.6794281
4: -92.5861588, 57.3921318, -96.6807480, 60.2719307, -152.8580933, 154.0728760
5: -71.8089142, 58.6708832, -75.2192383, 61.3835411, -133.1924438, 133.8901215
6: -73.9255219, 68.6816177, -77.3582916, 71.9089050, -145.8343964, 146.0399017
7: -78.4739304, 66.8039246, -82.1314621, 69.8784637, -148.3523865, 148.9353485
8: -92.7796097, 64.0383682, -97.2264862, 67.1470490, -159.9266510, 161.2648315
9: -75.2036514, 66.6150513, -78.6092148, 69.8383255, -145.0419769, 145.2242432

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6406068, upper bound: 154.6474466
time: 15.01 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6406068, upper bound: 154.7086554
time: 20.49 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -74.7671814, 58.1178017, -67.8828430, 52.4065094, -127.1736908, 126.0006409
1: -59.2831345, 52.9471054, -53.1625824, 48.0608902, -107.3440170, 106.1096878
2: -79.9682083, 55.1087112, -72.1232681, 50.3659096, -130.3340912, 127.2319794
3: -84.5132294, 47.0102005, -76.2197495, 42.9050865, -127.4183121, 123.2299500
4: -89.2509308, 54.3566170, -82.9525833, 47.5647964, -136.8157349, 137.3091888
5: -68.5117035, 56.1230927, -61.4994240, 50.8658485, -119.3775482, 117.6225128
6: -70.9274292, 65.7184906, -64.8103485, 59.5855904, -130.5130157, 130.5288391
7: -75.1480789, 64.0087357, -68.3062515, 58.2937508, -133.4418335, 132.3149872
8: -88.7273941, 61.2432671, -80.2249451, 55.3250847, -144.0524750, 141.4682159
9: -72.2363815, 63.4594116, -66.3222809, 56.6946487, -128.9310150, 129.7816925

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6386053, upper bound: 154.6320285
time: 13.67 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6392308, upper bound: 154.6324711
time: 12.90 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -82.4793930, 64.2899170, -83.2981796, 64.9178848, -147.3972778, 147.5880890
1: -65.8094025, 58.4177246, -66.4947433, 59.0004311, -124.8098297, 124.9124451
2: -88.5034561, 60.5652313, -89.3745728, 61.1455612, -149.6490173, 149.9398041
3: -93.5097885, 51.7365265, -94.4350815, 52.2653465, -145.7751312, 146.1716003
4: -97.4018021, 60.9177780, -98.3510666, 61.5330467, -158.9348450, 159.2688446
5: -75.9326935, 61.9275970, -76.6759415, 62.5312157, -138.4639130, 138.6035461
6: -77.9988632, 72.5447617, -78.7867508, 73.2695007, -151.2683563, 151.3314819
7: -82.8456421, 70.4815216, -83.6712494, 71.1681671, -154.0137482, 154.1527405
8: -98.0842896, 67.7380829, -99.0993576, 68.4526215, -166.5368805, 166.8374023
9: -79.2458572, 70.5169907, -80.0219803, 71.2244644, -150.4703217, 150.5389709

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6433765, upper bound: 154.6498743
time: 9.91 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6433765, upper bound: 154.7124971
time: 14.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.71 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.71
Output dim: 4, lower bound: -154.6359814, upper bound: 154.6294650
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.71
Output dim: 4, lower bound: -154.6370444, upper bound: 154.6303505
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.71
Output dim: 4, lower bound: -154.6406068, upper bound: 154.6474466
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.71
Output dim: 4, lower bound: -154.6406068, upper bound: 154.7086554
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.71
Output dim: 4, lower bound: -154.6386053, upper bound: 154.6320285
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.71
Output dim: 4, lower bound: -154.6392308, upper bound: 154.6324711
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.71
Output dim: 4, lower bound: -154.6433765, upper bound: 154.6498743
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.71
Output dim: 4, lower bound: -154.6433765, upper bound: 154.7124971

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -61.8878632, 47.9617233, -63.7555161, 49.1437988, -111.0316391, 111.7172165
1: -48.6519966, 43.8407211, -49.7377586, 45.1378365, -93.7898331, 93.5784760
2: -65.9121628, 45.9265213, -67.6131592, 47.4302673, -113.3424301, 113.5396805
3: -69.7016907, 39.0275993, -71.4529114, 40.3523598, -110.0540314, 110.4805145
4: -75.0527725, 44.0806198, -78.5099487, 44.2093582, -119.2621231, 122.5905685
5: -56.3591232, 46.5311012, -57.5908890, 47.7866173, -104.1457062, 104.1219864
6: -58.9704742, 54.3771553, -60.9961395, 55.9561920, -114.9266586, 115.3732910
7: -62.3103333, 53.2145920, -64.1977234, 54.8354492, -117.1457825, 117.4123001
8: -73.2653885, 50.5017357, -75.2610016, 51.8871460, -125.1525192, 125.7627411
9: -60.3227921, 52.1073494, -62.5294113, 53.0360832, -113.3588715, 114.6367645

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6296025, upper bound: 154.6218482
time: 19.42 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6338123, upper bound: 154.6271081
time: 12.97 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -65.2767639, 50.6123085, -64.4043198, 49.6517944, -114.9285431, 115.0166321
1: -51.4015121, 46.2345848, -50.2621231, 45.5943069, -96.9958191, 96.4967041
2: -69.5957947, 48.3620720, -68.3184891, 47.8954086, -117.4912033, 116.6805573
3: -73.5629578, 41.1247902, -72.1948013, 40.7543793, -114.3173370, 113.3195877
4: -78.9233551, 46.6684647, -79.2230759, 44.7175179, -123.6408691, 125.8915405
5: -59.5259628, 49.0552788, -58.1995621, 48.2683144, -107.7942734, 107.2548370
6: -62.1407204, 57.3572083, -61.5985489, 56.5232315, -118.6639557, 118.9557571
7: -65.6997833, 56.0698242, -64.8404083, 55.3792419, -121.0790253, 120.9102325
8: -77.3024292, 53.2947426, -76.0335922, 52.4218140, -129.7242432, 129.3283234
9: -63.5127029, 55.0309677, -63.1301231, 53.5982018, -117.1109009, 118.1610794

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6306999, upper bound: 154.6225141
time: 12.27 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6348608, upper bound: 154.6279742
time: 13.02 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -61.0134735, 46.9926720, -81.7621460, 63.7012024, -124.7146759, 128.7548218
1: -47.4360580, 43.2015343, -65.1951294, 57.9080200, -105.3440781, 108.3966599
2: -64.6352997, 45.4873924, -87.6928787, 60.0546951, -124.6899796, 133.1802521
3: -68.2957535, 38.6569023, -92.6549377, 51.3104820, -119.6062317, 131.3118286
4: -75.4870224, 41.9992981, -96.6807480, 60.2719307, -135.7589569, 138.6800385
5: -55.0194321, 45.7228470, -75.2192383, 61.3835411, -116.4029694, 120.9420853
6: -58.4250526, 53.5161018, -77.3582916, 71.9089050, -130.3339386, 130.8743896
7: -61.4343147, 52.4935989, -82.1314621, 69.8784637, -131.3127747, 134.6250458
8: -71.8887329, 49.5572662, -97.2264862, 67.1470490, -139.0357819, 146.7837524
9: -59.9705887, 50.5815125, -78.6092148, 69.8383255, -129.8089142, 129.1907043

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6209757, upper bound: 154.6369888
time: 9.56 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6204433, upper bound: 154.6374724
time: 10.59 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -75.6670303, 58.8925705, -81.7621460, 63.7012024, -139.3682251, 140.6547241
1: -60.0846748, 53.5883102, -65.1951294, 57.9080200, -117.9926910, 118.7834396
2: -81.0385742, 55.7261200, -87.6928787, 60.0546951, -141.0932617, 143.4190063
3: -85.6200790, 47.5263901, -92.6549377, 51.3104820, -136.9305573, 140.1813354
4: -89.9919891, 55.3355865, -96.6807480, 60.2719307, -150.2639160, 152.0163269
5: -69.4646378, 56.8362579, -75.2192383, 61.3835411, -130.8481750, 132.0554962
6: -71.6836777, 66.5193253, -77.3582916, 71.9089050, -143.5925751, 143.8776245
7: -76.0339203, 64.7512207, -82.1314621, 69.8784637, -145.9123840, 146.8826599
8: -89.8267441, 61.9960556, -97.2264862, 67.1470490, -156.9737854, 159.2225342
9: -72.9766541, 64.3889694, -78.6092148, 69.8383255, -142.8149719, 142.9981689

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6209757, upper bound: 154.7035861
time: 17.08 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6204433, upper bound: 154.7057054
time: 10.83 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -66.2269440, 51.3867760, -65.1983643, 50.2845688, -116.5115128, 116.5851364
1: -52.2849274, 46.9123268, -50.9579468, 46.1652527, -98.4501724, 97.8702698
2: -70.6469193, 49.0150452, -69.1893921, 48.4560661, -119.1029816, 118.2044373
3: -74.7003708, 41.7214622, -73.1233597, 41.2456818, -115.9460449, 114.8448181
4: -79.8349533, 47.5812569, -80.0638657, 45.3923683, -125.2273254, 127.6451187
5: -60.4599419, 49.7802315, -58.9578247, 48.8679504, -109.3278961, 108.7380524
6: -63.0175972, 58.2145996, -62.3370895, 57.2325592, -120.2501526, 120.5516739
7: -66.6597519, 56.8710670, -65.6441650, 56.0521049, -122.7118454, 122.5152283
8: -78.5156326, 54.1742859, -77.0179520, 53.1040993, -131.6197357, 131.1921997
9: -64.3496780, 55.9776611, -63.8597565, 54.3313942, -118.6810608, 119.8374100

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6317973, upper bound: 154.6242144
time: 12.58 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6365295, upper bound: 154.6296728
time: 11.93 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -69.4783096, 53.9245148, -65.8203278, 50.7714272, -120.2497253, 119.7448273
1: -54.9107895, 49.2087402, -51.4591942, 46.6033592, -101.5141373, 100.6679382
2: -74.1783752, 51.3524361, -69.8662872, 48.9011536, -123.0795288, 121.2187195
3: -78.3993301, 43.7311592, -73.8348236, 41.6303520, -120.0296783, 117.5659790
4: -83.5614548, 50.0532494, -80.7492981, 45.8810387, -129.4424896, 130.8025360
5: -63.4862900, 52.1956863, -59.5400810, 49.3299255, -112.8162155, 111.7357635
6: -66.0569382, 61.0731277, -62.9135628, 57.7754593, -123.8323898, 123.9866943
7: -69.9044342, 59.6058273, -66.2609863, 56.5729675, -126.4774017, 125.8668137
8: -82.3925400, 56.8532257, -77.7576218, 53.6175385, -136.0100708, 134.6108398
9: -67.4082565, 58.7691689, -64.4363403, 54.8691711, -122.2774277, 123.2054901

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6326227, upper bound: 154.6245952
time: 12.89 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6370874, upper bound: 154.6300531
time: 14.48 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -64.8345795, 50.0017662, -83.2981796, 64.9178848, -129.7524719, 133.2999420
1: -50.6013718, 45.8979492, -66.4947433, 59.0004311, -109.6018066, 112.3926849
2: -68.7912064, 48.2071762, -89.3745728, 61.1455612, -129.9367676, 137.5817413
3: -72.6958542, 41.0185890, -94.4350815, 52.2653465, -124.9611893, 135.4536591
4: -79.6702118, 45.0614700, -98.3510666, 61.5330467, -141.2032623, 143.4125366
5: -58.6162605, 48.5807838, -76.6759415, 62.5312157, -121.1474686, 125.2567291
6: -61.9852524, 56.8900986, -78.7867508, 73.2695007, -135.2547455, 135.6768494
7: -65.2527542, 55.7268562, -83.6712494, 71.1681671, -136.4208832, 139.3981018
8: -76.5235519, 52.7604256, -99.0993576, 68.4526215, -144.9761200, 151.8597870
9: -63.5092888, 53.9645233, -80.0219803, 71.2244644, -134.7337494, 133.9865112

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6230221, upper bound: 154.6386053
time: 9.15 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6225817, upper bound: 154.6392308
time: 11.09 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -80.0182724, 62.3286095, -83.2981796, 64.9178848, -144.9361572, 145.6267853
1: -63.7334671, 56.6745224, -66.4947433, 59.0004311, -122.7339020, 123.1692657
2: -85.7877960, 58.8206749, -89.3745728, 61.1455612, -146.9333496, 148.1952057
3: -90.6423111, 50.2274284, -94.4350815, 52.2653465, -142.9076538, 144.6625061
4: -94.7824707, 58.8527756, -98.3510666, 61.5330467, -156.3155212, 157.2038422
5: -73.5729752, 60.0822754, -76.6759415, 62.5312157, -136.1041870, 136.7582092
6: -75.7399292, 70.3668365, -78.7867508, 73.2695007, -149.0094299, 149.1535950
7: -80.3863068, 68.4134293, -83.6712494, 71.1681671, -151.5543976, 152.0846863
8: -95.1086044, 65.6820602, -99.0993576, 68.4526215, -163.5612030, 164.7814026
9: -77.0006256, 68.2752304, -80.0219803, 71.2244644, -148.2250519, 148.2972107

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6230221, upper bound: 154.6236331
time: 9.91 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6225817, upper bound: 154.7090299
time: 12.20 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.62 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6296025, upper bound: 154.6218482
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6338123, upper bound: 154.6271081
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6306999, upper bound: 154.6225141
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6348608, upper bound: 154.6279742
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6209757, upper bound: 154.6369888
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6204433, upper bound: 154.6374724
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6209757, upper bound: 154.7035861
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6204433, upper bound: 154.7057054
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6317973, upper bound: 154.6242144
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6365295, upper bound: 154.6296728
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6326227, upper bound: 154.6245952
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6370874, upper bound: 154.6300531
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6230221, upper bound: 154.6386053
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6225817, upper bound: 154.6392308
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6230221, upper bound: 154.6236331
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 4, lower bound: -154.6225817, upper bound: 154.7090299

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -56.5128059, 43.7232399, -51.1164665, 39.2168045, -95.7296066, 94.8397064
1: -44.2237358, 40.0636177, -39.4164734, 36.3016014, -80.5253372, 79.4800720
2: -60.0498238, 42.1138649, -53.8930664, 38.4809341, -98.5307541, 96.0069199
3: -63.4787140, 35.7133713, -56.8229675, 32.5830612, -96.0617676, 92.5363388
4: -69.2525711, 39.7273178, -64.8519058, 34.0493622, -103.3019333, 104.5792236
5: -51.2370491, 42.5046539, -45.5617714, 38.3314247, -89.5684738, 88.0664215
6: -54.0410309, 49.6479225, -49.4109993, 44.8604546, -98.9014893, 99.0589218
7: -56.9263306, 48.7037277, -51.5375214, 44.2155571, -101.1418915, 100.2412262
8: -66.8801117, 46.1288071, -60.3102036, 41.7087517, -108.5888672, 106.4390106
9: -55.4032288, 47.3140564, -50.9770813, 41.8577919, -97.2610092, 98.2911377

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6288152, upper bound: 154.6208644
time: 12.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6276745, upper bound: 154.6202176
time: 14.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -59.9422150, 46.4251213, -58.6226006, 45.1034775, -105.0456924, 105.0477142
1: -47.0392265, 42.4739571, -45.5316544, 41.5459671, -88.5851898, 88.0056152
2: -63.7861710, 44.5527000, -62.0249977, 43.8005180, -107.5866776, 106.5776978
3: -67.4506226, 37.8250237, -65.5169296, 37.2013893, -104.6520081, 103.3419266
4: -72.9666290, 42.4911003, -73.0097427, 40.0424080, -113.0090332, 115.5008392
5: -54.5004044, 45.0741196, -52.6993942, 43.9399834, -98.4403839, 97.7735138
6: -57.1906433, 52.6638298, -56.2980156, 51.4492264, -108.6398697, 108.9618454
7: -60.3701859, 51.5873947, -59.0825386, 50.5297394, -110.8999252, 110.6699219
8: -70.9524460, 48.9082985, -69.1574402, 47.7163010, -118.6687469, 118.0657349
9: -58.5494385, 50.3649902, -57.8496094, 48.4782333, -107.0276718, 108.2145996

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6326062, upper bound: 154.6257520
time: 12.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6318290, upper bound: 154.6252501
time: 11.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -59.8856010, 46.3599052, -51.7575150, 39.7175064, -99.6030807, 98.1174164
1: -46.9433670, 42.4408264, -39.9283638, 36.7478943, -83.6912613, 82.3691864
2: -63.7098961, 44.5436935, -54.5870285, 38.9411087, -102.6510010, 99.1307220
3: -67.3161011, 37.7886391, -57.5527649, 32.9765778, -100.2926636, 95.3414001
4: -73.1116104, 42.2904282, -65.5632706, 34.5481339, -107.6597443, 107.8536987
5: -54.3803940, 45.0180702, -46.1616936, 38.8066444, -93.1870422, 91.1797485
6: -57.1986694, 52.6066742, -50.0078125, 45.4180374, -102.6167068, 102.6144867
7: -60.3007965, 51.5491943, -52.1744232, 44.7533417, -105.0541382, 103.7236099
8: -70.9085083, 48.9001465, -61.0705299, 42.2317505, -113.1402588, 109.9706726
9: -58.5856133, 50.2091293, -51.5717697, 42.4070206, -100.9926300, 101.7808990

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6300249, upper bound: 154.6217242
time: 13.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6290180, upper bound: 154.6211283
time: 14.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -63.3090096, 49.0568962, -59.2474174, 45.5930710, -108.9020844, 108.3043137
1: -49.7706757, 44.8515434, -46.0316086, 41.9843254, -91.7549973, 90.8831482
2: -67.4429245, 46.9729958, -62.7021408, 44.2505569, -111.6934814, 109.6751328
3: -71.2863617, 39.9083710, -66.2316284, 37.5866699, -108.8730087, 106.1399994
4: -76.8128586, 45.0622673, -73.7026596, 40.5293846, -117.3422394, 118.7649231
5: -57.6465683, 47.5812645, -53.2856636, 44.4045525, -102.0511017, 100.8669281
6: -60.3410301, 55.6227074, -56.8814926, 51.9929962, -112.3340149, 112.5041962
7: -63.7367630, 54.4242592, -59.7040062, 51.0536041, -114.7903671, 114.1282654
8: -74.9641418, 51.6805382, -69.9009094, 48.2288780, -123.1930237, 121.5814438
9: -61.7183304, 53.2690315, -58.4288902, 49.0172157, -110.7355499, 111.6979218

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6337689, upper bound: 154.6265985
time: 14.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6329516, upper bound: 154.6261106
time: 11.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -58.5494003, 45.0480499, -73.0024872, 56.7920341, -115.3414307, 118.0505295
1: -45.4391823, 41.4753990, -58.0022850, 51.7124443, -97.1516190, 99.4776840
2: -61.9553642, 43.7310715, -78.1278229, 53.8006477, -115.7560120, 121.8588867
3: -65.4566650, 37.1347427, -82.5811539, 45.8843536, -111.3410034, 119.7158966
4: -72.8346634, 40.0183754, -87.0247192, 53.3223572, -126.1570206, 127.0430679
5: -52.6853104, 43.8911743, -66.9576797, 54.8823776, -107.5676727, 110.8488541
6: -56.1520309, 51.3619919, -69.2439957, 64.2107620, -120.3627625, 120.6059647
7: -58.9924507, 50.4327011, -73.4284286, 62.5522614, -121.5447083, 123.8611298
8: -68.9544525, 47.5448227, -86.7415466, 59.8907623, -128.8451843, 134.2863770
9: -57.7121696, 48.4354019, -70.5244217, 62.1467896, -119.8589554, 118.9598236

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6279901, upper bound: 154.6347553
time: 13.19 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6279235, upper bound: 154.6347553
time: 12.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -59.1368980, 45.5081940, -76.4984589, 59.5279541, -118.6648560, 122.0066528
1: -45.9105606, 41.8871346, -60.8425293, 54.1873093, -100.0978622, 102.7296600
2: -62.5924225, 44.1525002, -81.9290848, 56.3119545, -118.9043732, 126.0815887
3: -66.1300201, 37.4983330, -86.5659866, 48.0477905, -114.1778107, 124.0643082
4: -73.4788284, 40.4821854, -91.0030975, 56.0036774, -129.4824982, 131.4852905
5: -53.2386551, 44.3276291, -70.2222977, 57.4787903, -110.7174301, 114.5499115
6: -56.6985054, 51.8735809, -72.5108719, 67.2830353, -123.9815063, 124.3844528
7: -59.5756035, 50.9252281, -76.9144974, 65.4919510, -125.0675507, 127.8397141
8: -69.6510925, 48.0258179, -90.9142990, 62.7774506, -132.4285278, 138.9401245
9: -58.2551422, 48.9424133, -73.8018036, 65.1672287, -123.4223709, 122.7442169

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6285091, upper bound: 154.6354519
time: 13.18 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6284189, upper bound: 154.6354519
time: 13.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -72.6428452, 56.4995461, -73.0024872, 56.7920341, -129.4348755, 129.5020294
1: -57.5921707, 51.4508095, -58.0022850, 51.7124443, -109.3046112, 109.4530792
2: -77.7278519, 53.5720367, -78.1278229, 53.8006477, -131.5284882, 131.6998596
3: -82.1326828, 45.6511154, -82.5811539, 45.8843536, -128.0170288, 128.2322693
4: -86.7100601, 52.8956909, -87.0247192, 53.3223572, -140.0323944, 139.9203949
5: -66.6006622, 54.5933685, -66.9576797, 54.8823776, -121.4830399, 121.5510483
6: -68.8923492, 63.8626518, -69.2439957, 64.2107620, -133.1031036, 133.1066437
7: -73.0346985, 62.2294312, -73.4284286, 62.5522614, -135.5869598, 135.6578674
8: -86.2003479, 59.4814453, -86.7415466, 59.8907623, -146.0911102, 146.2229614
9: -70.2059021, 61.7147293, -70.5244217, 62.1467896, -132.3526917, 132.2391357

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6991621, upper bound: 154.7000002
time: 11.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7018055, upper bound: 154.7023688
time: 12.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.52 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6288152, upper bound: 154.6208644
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6276745, upper bound: 154.6202176
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6326062, upper bound: 154.6257520
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6318290, upper bound: 154.6252501
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6300249, upper bound: 154.6217242
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6290180, upper bound: 154.6211283
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6337689, upper bound: 154.6265985
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6329516, upper bound: 154.6261106
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6279901, upper bound: 154.6347553
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6279235, upper bound: 154.6347553
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6285091, upper bound: 154.6354519
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6284189, upper bound: 154.6354519
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.6991621, upper bound: 154.7000002
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 4, lower bound: -154.7018055, upper bound: 154.7023688
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 4, lower bound: -154.6204433, upper bound: 154.7057054
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 4, lower bound: -154.6317973, upper bound: 154.6242144
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 4, lower bound: -154.6365295, upper bound: 154.6296728
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 4, lower bound: -154.6326227, upper bound: 154.6245952
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 4, lower bound: -154.6370874, upper bound: 154.6300531
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 4, lower bound: -154.6230221, upper bound: 154.6386053
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 4, lower bound: -154.6225817, upper bound: 154.6392308
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 4, lower bound: -154.6230221, upper bound: 154.6236331
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 4, lower bound: -154.6225817, upper bound: 154.7090299
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=168.67294311523438
rel_dist={4: [-154.7150558205383, 154.7150558205383]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7088952, upper bound: 154.7092625
time: 12.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7136419, upper bound: 154.7136419
time: 12.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 25.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 25.03
Output dim: 4, lower bound: -154.7088952, upper bound: 154.7092625
IS_A2, status: Status.UNKNOWN, split count: 1, time: 25.03
Output dim: 4, lower bound: -154.7136419, upper bound: 154.7136419

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -78.1110840, 60.8416214, -80.9900818, 63.1061935, -141.2172852, 141.8316956
1: -62.1475143, 55.3199234, -64.5430222, 57.3557777, -119.5032959, 119.8629456
2: -83.7355194, 57.4565086, -86.8694229, 59.5048447, -143.2403564, 144.3258972
3: -88.4660645, 49.0245056, -91.7797928, 50.8110199, -139.2770691, 140.8042908
4: -92.5861588, 57.3921318, -95.7809448, 59.6898537, -152.2759857, 153.1730652
5: -71.8089142, 58.6708832, -74.5161362, 60.8155441, -132.6244507, 133.1870117
6: -73.9255219, 68.6816177, -76.6101990, 71.2233658, -145.1488953, 145.2918091
7: -78.4739304, 66.8039246, -81.3494110, 69.2303162, -147.7042542, 148.1533203
8: -92.7796097, 64.0383682, -96.2633514, 66.4660873, -159.2456970, 160.3017273
9: -75.2036514, 66.6150513, -77.8768539, 69.1669159, -144.3705750, 144.4918976

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6365569, upper bound: 154.6343402
time: 13.43 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7074127, upper bound: 154.7078311
time: 17.52 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -82.4793930, 64.2899170, -83.8287354, 65.3552475, -147.8346405, 148.1186523
1: -65.8094025, 58.4177246, -66.9460602, 59.3750153, -125.1844025, 125.3637848
2: -88.5034561, 60.5652313, -89.9800034, 61.5216789, -150.0251312, 150.5452271
3: -93.5097885, 51.7365265, -95.0702286, 52.5747566, -146.0845490, 146.8067627
4: -97.4018021, 60.9177780, -98.8697968, 62.0207825, -159.4225616, 159.7875671
5: -75.9326935, 61.9275970, -77.2097855, 62.9347534, -138.8674469, 139.1373901
6: -77.9988632, 72.5447617, -79.2523346, 73.7396545, -151.7384949, 151.7970428
7: -82.8456421, 70.4815216, -84.1976318, 71.6152344, -154.4608459, 154.6791534
8: -98.0842896, 67.7380829, -99.7269821, 68.8781738, -166.9624481, 167.4650269
9: -79.2458572, 70.5169907, -80.4890594, 71.7309418, -150.9768066, 151.0060120

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6392872, upper bound: 154.6369266
time: 17.04 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7124052, upper bound: 154.7124051
time: 19.11 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 37.64 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 37.64
Output dim: 4, lower bound: -154.6365569, upper bound: 154.6343402
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 37.64
Output dim: 4, lower bound: -154.7074127, upper bound: 154.7078311
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 37.64
Output dim: 4, lower bound: -154.6392872, upper bound: 154.6369266
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 37.64
Output dim: 4, lower bound: -154.7124052, upper bound: 154.7124051

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -63.3324623, 49.0143051, -63.4645767, 48.9137650, -112.2462311, 112.4788818
1: -49.6602936, 44.8417053, -49.4433746, 44.9225349, -94.5828247, 94.2850723
2: -67.3877563, 47.0019417, -67.2924118, 47.2289581, -114.6167145, 114.2943573
3: -71.2519455, 39.9762497, -71.1080246, 40.1709518, -111.4228897, 111.0842743
4: -76.9982758, 44.7998962, -78.1916809, 43.9398613, -120.9381256, 122.9915771
5: -57.5853767, 47.5234871, -57.3172913, 47.5528603, -105.1382370, 104.8407745
6: -60.3817635, 55.6053734, -60.7072716, 55.6770821, -116.0588455, 116.3126450
7: -63.7265663, 54.4044495, -63.8774376, 54.5688858, -118.2954559, 118.2818909
8: -74.8855286, 51.5996552, -74.8480072, 51.6005402, -126.4860687, 126.4476624
9: -61.7639961, 53.1165771, -62.2460594, 52.7323341, -114.4963303, 115.3626328

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6281209, upper bound: 154.6261385
time: 13.38 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6335859, upper bound: 154.6311730
time: 11.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -76.8044510, 59.7995987, -78.5306854, 61.1450424, -137.9494934, 138.3302765
1: -61.0443840, 54.3939285, -62.4675407, 55.6126862, -116.6570740, 116.8614655
2: -82.2935562, 56.5312576, -84.1553726, 57.7620430, -140.0555725, 140.6866302
3: -86.9442139, 48.2234688, -88.9142380, 49.3028717, -136.2470703, 137.1377106
4: -91.1985397, 56.2928886, -93.1653290, 57.6235847, -148.8221283, 149.4582214
5: -70.5555267, 57.6897392, -72.1574783, 58.9693947, -129.5249176, 129.8471985
6: -72.7268066, 67.5255051, -74.3530731, 69.0470734, -141.7738800, 141.8785706
7: -77.1691742, 65.7062607, -78.8925781, 67.1635437, -144.3327179, 144.5988464
8: -91.2008743, 62.9463043, -93.2901917, 64.4106598, -155.6115417, 156.2364960
9: -74.0126877, 65.4247437, -75.6336670, 66.9270325, -140.9397125, 141.0583954

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7012987, upper bound: 154.7018724
time: 10.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7046296, upper bound: 154.7049207
time: 11.57 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -67.5621185, 52.3517303, -66.0743103, 50.9781303, -118.5402527, 118.4260406
1: -53.1972313, 47.8388557, -51.6430588, 46.7774200, -99.9746399, 99.4819031
2: -72.0013657, 50.0144577, -70.1449814, 49.0842552, -121.0856171, 120.1594315
3: -76.1222153, 42.5976830, -74.1291656, 41.7854347, -117.9076462, 116.7268524
4: -81.6531067, 48.2107620, -81.0045624, 46.0790558, -127.7321396, 129.2153320
5: -61.5749207, 50.6877403, -59.7881775, 49.5100594, -111.0849609, 110.4759216
6: -64.3240204, 59.3445244, -63.1336517, 57.9861031, -122.3101196, 122.4781799
7: -67.9596786, 57.9689636, -66.4948044, 56.7709312, -124.7306061, 124.4637680
8: -80.0060120, 55.1819115, -78.0277328, 53.8017426, -133.8077545, 133.2096405
9: -65.6887436, 56.8811455, -64.6530991, 55.0735626, -120.7623062, 121.5342407

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6300371, upper bound: 154.6279019
time: 12.57 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6368368, upper bound: 154.6342553
time: 11.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -81.1622620, 63.2402649, -81.3645859, 63.3916702, -144.5539246, 144.6048584
1: -64.6984406, 57.4846268, -64.8670425, 57.6293755, -122.3278198, 122.3516464
2: -87.0499725, 59.6313171, -87.2605820, 59.7746849, -146.8246613, 146.8919067
3: -91.9753342, 50.9289207, -92.1992493, 51.0639038, -143.0392303, 143.1281738
4: -95.9992294, 59.8131981, -96.2467804, 59.9538689, -155.9530945, 156.0599670
5: -74.6699219, 60.9400063, -74.8470001, 61.0876274, -135.7575378, 135.7870026
6: -76.7899170, 71.3790359, -76.9908295, 71.5586243, -148.3485413, 148.3698730
7: -81.5293350, 69.3747406, -81.7350311, 69.5445099, -151.0738525, 151.1097717
8: -96.4914246, 66.6374283, -96.7468262, 66.8192062, -163.3106232, 163.3842163
9: -78.0439835, 69.3172531, -78.2411346, 69.4860992, -147.5300903, 147.5583649

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7075059, upper bound: 154.7076087
time: 9.69 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7088722, upper bound: 154.7088710
time: 14.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.33 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.33
Output dim: 4, lower bound: -154.6281209, upper bound: 154.6261385
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.33
Output dim: 4, lower bound: -154.6335859, upper bound: 154.6311730
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.33
Output dim: 4, lower bound: -154.7012987, upper bound: 154.7018724
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.33
Output dim: 4, lower bound: -154.7046296, upper bound: 154.7049207
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.33
Output dim: 4, lower bound: -154.6300371, upper bound: 154.6279019
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.33
Output dim: 4, lower bound: -154.6368368, upper bound: 154.6342553
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.33
Output dim: 4, lower bound: -154.7075059, upper bound: 154.7076087
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.33
Output dim: 4, lower bound: -154.7088722, upper bound: 154.7088710

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -50.2030945, 38.6876068, -54.4244804, 41.8077202, -92.0108185, 93.1120834
1: -38.9235687, 35.6395111, -42.0623665, 38.5933113, -77.5168762, 77.7018738
2: -53.0967064, 37.6889343, -57.4670029, 40.8265305, -93.9232178, 95.1559372
3: -56.0649681, 31.8897686, -60.6314163, 34.5975456, -90.6625137, 92.5211868
4: -62.7525978, 34.2711258, -68.4458771, 36.6419640, -99.3945618, 102.7169952
5: -45.1088943, 37.6970253, -48.7001038, 40.7811623, -85.8900604, 86.3971252
6: -48.3368149, 44.0567017, -52.4319839, 47.7309570, -96.0677719, 96.4886780
7: -50.5724564, 43.3616486, -54.8245010, 46.9684639, -97.5409241, 98.1861420
8: -59.2893753, 40.9593124, -64.1567383, 44.3193626, -103.6087341, 105.1160507
9: -49.7217026, 41.4917336, -53.9955101, 44.7290993, -94.4507980, 95.4872437

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6258305, upper bound: 154.6239695
time: 12.91 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6258289, upper bound: 154.6239270
time: 13.08 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -58.0042267, 44.8117599, -59.6632233, 45.9224434, -103.9266663, 104.4749756
1: -45.2716675, 41.0998573, -46.3348732, 42.2635651, -87.5352325, 87.4347229
2: -61.5726585, 43.2363472, -63.1530991, 44.5461617, -106.1188202, 106.3894424
3: -65.0895233, 36.6944847, -66.7087479, 37.8284607, -102.9179840, 103.4032288
4: -71.2742310, 40.4735031, -74.1215515, 40.8552284, -112.1294556, 114.5950546
5: -52.5117035, 43.5302544, -53.6955299, 44.7048950, -97.2165985, 97.2257690
6: -55.5057030, 50.9210625, -57.2332458, 52.3372955, -107.8430023, 108.1542969
7: -58.4108734, 49.9359818, -60.0928612, 51.3791466, -109.7900238, 110.0288239
8: -68.5443573, 47.2473640, -70.3338013, 48.5149269, -117.0592728, 117.5811615
9: -56.8899460, 48.3772812, -58.7855148, 49.3601341, -106.2500763, 107.1627960

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6312648, upper bound: 154.6290186
time: 12.23 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6312640, upper bound: 154.6289608
time: 13.55 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -67.9502945, 52.8160172, -73.0566940, 56.8129311, -124.7632294, 125.8727036
1: -53.7770920, 48.1336937, -57.9541855, 51.7444382, -105.5215302, 106.0878754
2: -72.6245117, 50.2081375, -78.1647263, 53.8663139, -126.4907990, 128.3728638
3: -76.7672501, 42.7360916, -82.5980377, 45.9103355, -122.6775818, 125.3340988
4: -81.4273758, 49.2738037, -87.2416611, 53.1993790, -134.6267548, 136.5154419
5: -62.2063980, 51.1128311, -66.9694672, 54.9090843, -117.1154785, 118.0822983
6: -64.5205612, 59.7434387, -69.3030777, 64.2419510, -128.7624969, 129.0465088
7: -68.3640671, 58.2991104, -73.4661255, 62.6010628, -130.9651184, 131.7652283
8: -80.6015701, 55.6129684, -86.7337418, 59.8611145, -140.4626770, 142.3467102
9: -65.8326035, 57.6536446, -70.6247787, 62.0837097, -127.9163132, 128.2784271

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6978697, upper bound: 154.6980033
time: 11.75 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7001373, upper bound: 154.7006081
time: 16.80 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -71.4816666, 55.5822906, -74.5280838, 57.9731636, -129.4548340, 130.1103821
1: -56.6496620, 50.6330376, -59.1595955, 52.7848091, -109.4344711, 109.7926331
2: -76.4682465, 52.7448654, -79.7736359, 54.9160461, -131.3842773, 132.5184937
3: -80.7968674, 44.9228401, -84.2886353, 46.8215141, -127.6183777, 129.2114716
4: -85.4468002, 51.9859695, -88.8463058, 54.3795319, -139.8263092, 140.8322601
5: -65.5099716, 53.7403717, -68.3602676, 56.0005417, -121.5105133, 122.1006317
6: -67.8240585, 62.8486290, -70.6674118, 65.5300217, -133.3540802, 133.5160370
7: -71.8944244, 61.2716751, -74.9267044, 63.8291550, -135.7235718, 136.1983795
8: -84.8213501, 58.5276299, -88.4921722, 61.0876617, -145.9090118, 147.0197754
9: -69.1472397, 60.7110710, -71.9784393, 63.3788757, -132.5261230, 132.6895142

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7000220, upper bound: 154.6999510
time: 12.62 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7034160, upper bound: 154.7037281
time: 13.54 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -54.0851746, 41.7451477, -56.7483673, 43.6414871, -97.7266541, 98.4935074
1: -42.1053352, 38.3655128, -43.9856606, 40.2259331, -82.3312683, 82.3511734
2: -57.3115768, 40.4529572, -59.9898834, 42.4739037, -99.7854767, 100.4428406
3: -60.5280991, 34.2868195, -63.3200378, 36.0385475, -96.5666504, 97.6068573
4: -67.0453415, 37.3616982, -70.9609528, 38.5302124, -105.5755539, 108.3226395
5: -48.7522888, 40.6069183, -50.9035797, 42.5204697, -91.2727509, 91.5104980
6: -51.9520416, 47.4765358, -54.5903702, 49.7801743, -101.7322006, 102.0669022
7: -54.4550934, 46.6504021, -57.1583977, 48.9319229, -103.3870087, 103.8087997
8: -64.0027771, 44.2036591, -66.9623337, 46.2362823, -110.2390594, 111.1659927
9: -53.3265991, 44.8985443, -56.1307449, 46.7820587, -100.1086578, 101.0292892

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6280101, upper bound: 154.6260584
time: 15.92 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6280101, upper bound: 154.6260305
time: 12.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -62.0950394, 48.0357628, -62.1706619, 47.9035759, -109.9986115, 110.2064209
1: -48.6769981, 43.9973488, -48.4172173, 44.0311813, -92.7081528, 92.4145660
2: -66.0286484, 46.1515083, -65.8898468, 46.3273315, -112.3559799, 112.0413361
3: -69.8003540, 39.2190323, -69.6085205, 39.3800926, -109.1804504, 108.8275528
4: -75.7696381, 43.7631683, -76.8300476, 42.8990898, -118.6687317, 120.5932159
5: -56.3610153, 46.5930786, -56.0703697, 46.5855484, -102.9465637, 102.6634521
6: -59.3186836, 54.5299950, -59.5668755, 54.5531425, -113.8718185, 114.0968552
7: -62.5079651, 53.3940201, -62.6040726, 53.4986343, -116.0065994, 115.9980774
8: -73.5133209, 50.7039871, -73.3940048, 50.6215858, -124.1349030, 124.0979919
9: -60.6946449, 52.0011330, -61.0921402, 51.5959625, -112.2906036, 113.0932770

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6348564, upper bound: 154.6324332
time: 17.96 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6348564, upper bound: 154.6323951
time: 14.52 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -72.4228821, 56.3504944, -75.9622498, 59.1155853, -131.5384674, 132.3127289
1: -57.5270309, 51.3052063, -60.4129524, 53.8109779, -111.3380127, 111.7181549
2: -77.5094223, 53.3908539, -81.3477325, 55.9287300, -133.4381409, 134.7385864
3: -81.9282837, 45.5151520, -85.9621277, 47.7162552, -129.6445312, 131.4772797
4: -86.3551331, 52.8893127, -90.3957825, 55.5916519, -141.9467773, 143.2850647
5: -66.4338531, 54.4549751, -69.7270508, 57.0808830, -123.5147400, 124.1820221
6: -68.6935654, 63.6990280, -72.0069427, 66.8152542, -135.5088196, 135.7059631
7: -72.8469543, 62.0656891, -76.3786926, 65.0393372, -137.8862610, 138.4443817
8: -86.0315704, 59.3985596, -90.2745361, 62.3308105, -148.3623810, 149.6730957
9: -69.9741974, 61.6512680, -73.2947159, 64.7063599, -134.6805573, 134.9459839

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7020483, upper bound: 154.7018104
time: 14.73 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7064688, upper bound: 154.7065572
time: 11.60 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -75.7704697, 58.9666748, -77.2806396, 60.1544647, -135.9249268, 136.2472992
1: -60.2386093, 53.6733170, -61.4887848, 54.7419052, -114.9804993, 115.1620941
2: -81.1459351, 55.7973671, -82.7875137, 56.8706207, -138.0165558, 138.5848846
3: -85.7411575, 47.5861893, -87.4761353, 48.5320358, -134.2731781, 135.0623016
4: -90.1790009, 55.4441185, -91.8366470, 56.6451988, -146.8242035, 147.2807465
5: -69.5530090, 56.9407272, -70.9708252, 58.0585709, -127.6115570, 127.9115295
6: -71.8232880, 66.6399765, -73.2286148, 67.9682617, -139.7915497, 139.8685913
7: -76.1855011, 64.8807602, -77.6866760, 66.1400986, -142.3255920, 142.5674438
8: -90.0249100, 62.1587410, -91.8474350, 63.4275551, -153.4524536, 154.0061798
9: -73.1172180, 64.5330200, -74.5088425, 65.8618317, -138.9790344, 139.0418549

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7033107, upper bound: 154.7029635
time: 10.61 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7077987, upper bound: 154.7077987
time: 9.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.33 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.6258305, upper bound: 154.6239695
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.6258289, upper bound: 154.6239270
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.6312648, upper bound: 154.6290186
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.6312640, upper bound: 154.6289608
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.6978697, upper bound: 154.6980033
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.7001373, upper bound: 154.7006081
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.7000220, upper bound: 154.6999510
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.7034160, upper bound: 154.7037281
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.6280101, upper bound: 154.6260584
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.6280101, upper bound: 154.6260305
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.6348564, upper bound: 154.6324332
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.6348564, upper bound: 154.6323951
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.7020483, upper bound: 154.7018104
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.7064688, upper bound: 154.7065572
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.7033107, upper bound: 154.7029635
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.33
Output dim: 4, lower bound: -154.7077987, upper bound: 154.7077987

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -48.2354240, 37.1673241, -51.6980515, 39.7044792, -87.9398956, 88.8653717
1: -37.3644981, 34.2628288, -39.9087029, 36.6894722, -74.0539627, 74.1715317
2: -50.9810028, 36.2554207, -54.5447502, 38.8450966, -89.8260880, 90.8001709
3: -53.8447647, 30.6763821, -57.5574341, 32.9176025, -86.7623672, 88.2338028
4: -60.4033966, 32.8416824, -65.1908798, 34.6705742, -95.0739670, 98.0325470
5: -43.3101120, 36.2256241, -46.2134361, 38.7445412, -82.0546417, 82.4390564
6: -46.4758911, 42.3367195, -49.8566017, 45.3533669, -91.8292542, 92.1933136
7: -48.5934372, 41.6883240, -52.0847168, 44.6535454, -93.2469788, 93.7730408
8: -56.9642181, 39.3652496, -60.9415321, 42.1161346, -99.0803528, 100.3067703
9: -47.8362923, 39.8443565, -51.3857460, 42.4529457, -90.2892303, 91.2301025

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6114505, upper bound: 154.6093026
time: 12.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6109564, upper bound: 154.6089197
time: 11.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -48.2288895, 37.1598396, -59.4573746, 45.6808548, -93.9097443, 96.6172180
1: -37.3503227, 34.2582626, -46.0468521, 42.0879974, -79.4383240, 80.3051147
2: -50.9705353, 36.2568970, -62.7924881, 44.5201607, -95.4906845, 99.0493774
3: -53.8269386, 30.6735744, -66.2946167, 37.7737427, -91.6006775, 96.9681931
4: -60.4235458, 32.8216705, -74.4430923, 40.1895981, -100.6131439, 107.2647629
5: -43.2959442, 36.2205811, -53.2581863, 44.5135727, -87.8095169, 89.4787521
6: -46.4732780, 42.3333740, -57.1681786, 52.1655312, -98.6388092, 99.5015488
7: -48.5876007, 41.6871223, -59.8829269, 51.2863503, -99.8739471, 101.5700455
8: -56.9571838, 39.3637772, -70.1293488, 48.4675179, -105.4246979, 109.4931107
9: -47.8418770, 39.8363571, -58.8757286, 48.9617805, -96.8036575, 98.7120743

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6114505, upper bound: 154.6092787
time: 12.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6109526, upper bound: 154.6088539
time: 11.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -55.9092178, 43.1942940, -56.8028564, 43.7181778, -99.6273956, 99.9971390
1: -43.6079597, 39.6286659, -44.0676041, 40.2592850, -83.8672485, 83.6962738
2: -59.3188095, 41.7091522, -60.0843964, 42.4602623, -101.7790680, 101.7935410
3: -62.7227249, 35.4009514, -63.4809113, 36.0601997, -98.7829132, 98.8818512
4: -68.7743149, 38.9514732, -70.7069855, 38.7885857, -107.5628967, 109.6584320
5: -50.6006241, 41.9620667, -51.0874062, 42.5696945, -93.1703186, 93.0494690
6: -53.5254021, 49.0904541, -54.5275192, 49.8404961, -103.3658981, 103.6179733
7: -56.3038559, 48.1552620, -57.2167473, 48.9485512, -105.2524109, 105.3719940
8: -66.0619125, 45.5399399, -66.9565582, 46.2014046, -112.2633057, 112.4964905
9: -54.8809891, 46.6196518, -56.0430374, 46.9705849, -101.8515778, 102.6626892

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6212403, upper bound: 154.6188834
time: 14.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6216519, upper bound: 154.6192446
time: 12.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -55.8214836, 43.1268730, -64.7456055, 49.8439407, -105.6654205, 107.8724823
1: -43.5293884, 39.5663071, -50.3934822, 45.8053894, -89.3347702, 89.9597931
2: -59.2218170, 41.6489868, -68.5577087, 48.2755814, -107.4973984, 110.2066727
3: -62.6144791, 35.3474045, -72.4248352, 41.0474319, -103.6619110, 107.7722168
4: -68.6925583, 38.8772583, -80.1308594, 44.4739418, -113.1664810, 119.0081177
5: -50.5149918, 41.8971977, -58.3169327, 48.4682884, -98.9832764, 100.2141113
6: -53.4445763, 49.0155106, -62.0118370, 56.8342896, -110.2788696, 111.0273438
7: -56.2157059, 48.0842819, -65.1972427, 55.7376480, -111.9533463, 113.2815170
8: -65.9594269, 45.4740257, -76.4110336, 52.7398491, -118.6992798, 121.8850555
9: -54.8059998, 46.5449562, -63.7107353, 53.6621208, -108.4681244, 110.2556915

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6167306, upper bound: 154.6141264
time: 15.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6162817, upper bound: 154.6137791
time: 13.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -58.1030273, 45.0494690, -59.2360153, 45.9168892, -104.0199127, 104.2854767
1: -45.6316299, 41.2050247, -46.5292778, 42.0204926, -87.6521072, 87.7342987
2: -61.8678474, 43.2367020, -63.0722046, 44.0803719, -105.9482193, 106.3089066
3: -65.3553696, 36.6436577, -66.5892410, 37.3633995, -102.7187653, 103.2328949
4: -70.8099747, 41.2802353, -72.2879639, 42.0312042, -112.8411560, 113.5681992
5: -52.8085861, 43.7445221, -53.8024940, 44.5838203, -97.3924103, 97.5470123
6: -55.4966621, 51.0653687, -56.6210175, 52.0588493, -107.5555115, 107.6863861
7: -58.5002441, 50.0460052, -59.6210251, 51.0212326, -109.5214615, 109.6670303
8: -68.9146500, 47.5793381, -70.3110352, 48.5895233, -117.5041733, 117.8903656
9: -56.8370018, 48.8398170, -57.9817810, 49.7328644, -106.5698700, 106.8215942

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6166465, upper bound: 154.6935556
time: 14.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6932085, upper bound: 154.6933766
time: 14.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -63.8378448, 49.5647125, -67.3826599, 52.3267593, -116.1646042, 116.9473572
1: -50.3635788, 45.2408981, -53.2446060, 47.7513733, -98.1149521, 98.4855042
2: -68.1200180, 47.3083076, -71.9514542, 49.8635254, -117.9835434, 119.2597656
3: -71.9990997, 40.1926651, -76.0196381, 42.3994331, -114.3985291, 116.2122955
4: -77.0221252, 45.9077415, -81.1556625, 48.5582886, -125.5803909, 127.0634003
5: -58.2748604, 48.0374908, -61.5488815, 50.6684494, -108.9432907, 109.5863724
6: -60.7616844, 56.1164131, -64.1119003, 59.2402191, -120.0019073, 120.2283020
7: -64.2586441, 54.8619690, -67.8030319, 57.8591957, -122.1178360, 122.6649933
8: -75.7058105, 52.2359619, -79.9698715, 55.2025414, -130.9083557, 132.2058258
9: -62.0882034, 53.9615822, -65.4602280, 56.9908943, -119.0791016, 119.4218063

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6961257, upper bound: 154.6962652
time: 12.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6957549, upper bound: 154.6961280
time: 14.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 28.08 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6114505, upper bound: 154.6093026
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6109564, upper bound: 154.6089197
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6114505, upper bound: 154.6092787
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6109526, upper bound: 154.6088539
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6212403, upper bound: 154.6188834
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6216519, upper bound: 154.6192446
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6167306, upper bound: 154.6141264
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6162817, upper bound: 154.6137791
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6166465, upper bound: 154.6935556
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6932085, upper bound: 154.6933766
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6961257, upper bound: 154.6962652
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.08
Output dim: 4, lower bound: -154.6957549, upper bound: 154.6961280
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.08
Output dim: 4, lower bound: -154.7000220, upper bound: 154.6999510
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.08
Output dim: 4, lower bound: -154.7034160, upper bound: 154.7037281
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.08
Output dim: 4, lower bound: -154.6280101, upper bound: 154.6260584
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.08
Output dim: 4, lower bound: -154.6280101, upper bound: 154.6260305
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.08
Output dim: 4, lower bound: -154.6348564, upper bound: 154.6324332
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.08
Output dim: 4, lower bound: -154.6348564, upper bound: 154.6323951
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.08
Output dim: 4, lower bound: -154.7020483, upper bound: 154.7018104
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.08
Output dim: 4, lower bound: -154.7064688, upper bound: 154.7065572
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.08
Output dim: 4, lower bound: -154.7033107, upper bound: 154.7029635
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.08
Output dim: 4, lower bound: -154.7077987, upper bound: 154.7077987
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=168.67294311523438
rel_dist={4: [-154.71496529634885, 154.71496529634885]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1834.73 seconds
