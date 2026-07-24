## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 195.3388952653
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-114.5843811, 90.7904587, -114.5843811, 90.7904587, -205.3747864, 205.3747864)
1: (-94.4460678, 79.3688889, -94.4460678, 79.3688889, -173.8149567, 173.8149567)
2: (-123.0286026, 78.4946442, -123.0286026, 78.4946442, -201.5232544, 201.5232544)
3: (-135.7945709, 68.0107193, -135.7945709, 68.0107193, -203.8052521, 203.8052521)
4: (-123.7332611, 94.1375732, -123.7332611, 94.1375732, -217.8708344, 217.8708344)
5: (-109.2941971, 85.3564529, -109.2941971, 85.3564529, -194.6506042, 194.6506042)
6: (-104.0020294, 101.6483536, -104.0020294, 101.6483536, -205.6503754, 205.6503754)
7: (-116.2566986, 91.5730362, -116.2566986, 91.5730362, -207.8297424, 207.8297424)
8: (-134.8109283, 92.3462830, -134.8109283, 92.3462830, -227.1571655, 227.1571655)
9: (-103.4061813, 102.4550095, -103.4061813, 102.4550095, -205.8611908, 205.8611908)

## BASE Result
execution time: IAR + LP analysis = 1.25 + 10.46 = 11.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -195.3645330, upper bound: 195.3645330


# Binary Search by BASE starts (time budget: 2688.29 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.82974243164062
rel_dist={7: [-195.364420828844, 195.364420828844]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=207.82974243164062
rel_dist={7: [-195.3643469135574, 195.36434691615057]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=207.82974243164062
rel_dist={7: [-195.36410562008308, 195.3641056191758]}

## Binary Search Result
Binary search time: 49.14 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2639.15 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3550413, upper bound: 195.3536772
time: 13.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3600031, upper bound: 195.3600031
time: 38.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 51.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 51.83
Output dim: 7, lower bound: -195.3550413, upper bound: 195.3536772
IS_A2, status: Status.UNKNOWN, split count: 1, time: 51.83
Output dim: 7, lower bound: -195.3600031, upper bound: 195.3600031

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -92.4082718, 73.1420441, -111.8682861, 88.6249847, -181.0332642, 185.0103149
1: -75.9085846, 63.7761002, -92.1604538, 77.4541168, -153.3627014, 155.9365540
2: -98.5751877, 62.5427284, -120.0300674, 76.5435791, -175.1187439, 182.5727997
3: -110.0107346, 54.3026886, -132.6379395, 66.3266068, -176.3373413, 186.9406128
4: -99.9523087, 75.8515472, -120.8181992, 91.8928833, -191.8451843, 196.6697388
5: -87.9972458, 68.5688782, -106.6816483, 83.3010941, -171.2983246, 175.2505035
6: -83.5991364, 82.1437836, -101.5032654, 99.2561264, -182.8552551, 183.6470490
7: -94.1258621, 72.6419296, -113.5413589, 89.2604828, -183.3863525, 186.1832886
8: -107.8815155, 73.8112030, -131.5102844, 90.0744705, -197.9559784, 205.3214874
9: -83.1070404, 82.4340668, -100.9216995, 99.9931488, -183.1001892, 183.3557739

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3517834, upper bound: 195.3517834
time: 8.63 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3517834, upper bound: 195.3536772
time: 10.05 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -104.9692230, 83.1579132, -114.5843811, 90.7904587, -195.7596588, 197.7422791
1: -86.3696899, 72.5907288, -94.4460678, 79.3688889, -165.7385559, 167.0367737
2: -112.4341278, 71.6278458, -123.0286026, 78.4946442, -190.9287415, 194.6564331
3: -124.5501099, 62.0677834, -135.7945709, 68.0107193, -192.5608215, 197.8623047
4: -113.4068680, 86.2109833, -123.7332611, 94.1375732, -207.5444336, 209.9442444
5: -100.0609283, 78.1082764, -109.2941971, 85.3564529, -185.4173584, 187.4024353
6: -95.1717758, 93.1705627, -104.0020294, 101.6483536, -196.8201294, 197.1725464
7: -106.6246414, 83.4336395, -116.2566986, 91.5730362, -198.1976776, 199.6903381
8: -123.1846008, 84.3752060, -134.8109283, 92.3462830, -215.5308685, 219.1861267
9: -94.6163712, 93.7728882, -103.4061813, 102.4550095, -197.0713806, 197.1790771

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3536772, upper bound: 195.3550413
time: 11.67 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3536772, upper bound: 195.3600031
time: 10.13 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.13
Output dim: 7, lower bound: -195.3517834, upper bound: 195.3517834
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.13
Output dim: 7, lower bound: -195.3517834, upper bound: 195.3536772
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.13
Output dim: 7, lower bound: -195.3536772, upper bound: 195.3550413
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.13
Output dim: 7, lower bound: -195.3536772, upper bound: 195.3600031

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -92.4082718, 73.1420441, -92.4082718, 73.1420441, -165.5502930, 165.5502930
1: -75.9085846, 63.7761002, -75.9085846, 63.7761002, -139.6846924, 139.6846924
2: -98.5751877, 62.5427284, -98.5751877, 62.5427284, -161.1179199, 161.1179199
3: -110.0107346, 54.3026886, -110.0107346, 54.3026886, -164.3134155, 164.3134155
4: -99.9523087, 75.8515472, -99.9523087, 75.8515472, -175.8038483, 175.8038483
5: -87.9972458, 68.5688782, -87.9972458, 68.5688782, -156.5661011, 156.5661011
6: -83.5991364, 82.1437836, -83.5991364, 82.1437836, -165.7429047, 165.7429199
7: -94.1258621, 72.6419296, -94.1258621, 72.6419296, -166.7677917, 166.7677917
8: -107.8815155, 73.8112030, -107.8815155, 73.8112030, -181.6927185, 181.6927185
9: -83.1070404, 82.4340668, -83.1070404, 82.4340668, -165.5411072, 165.5411072

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3467741, upper bound: 195.3472966
time: 10.93 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3461601, upper bound: 195.3461601
time: 8.06 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -92.4082718, 73.1420441, -104.9692230, 83.1579132, -175.5661774, 178.1112518
1: -75.9085846, 63.7761002, -86.3696899, 72.5907288, -148.4992981, 150.1457672
2: -98.5751877, 62.5427284, -112.4341278, 71.6278458, -170.2030029, 174.9768372
3: -110.0107346, 54.3026886, -124.5501099, 62.0677834, -172.0785217, 178.8527832
4: -99.9523087, 75.8515472, -113.4068680, 86.2109833, -186.1632690, 189.2584229
5: -87.9972458, 68.5688782, -100.0609283, 78.1082764, -166.1054993, 168.6297913
6: -83.5991364, 82.1437836, -95.1717758, 93.1705627, -176.7696533, 177.3155518
7: -94.1258621, 72.6419296, -106.6246414, 83.4336395, -177.5595093, 179.2665710
8: -107.8815155, 73.8112030, -123.1846008, 84.3752060, -192.2567139, 196.9958038
9: -83.1070404, 82.4340668, -94.6163712, 93.7728882, -176.8799286, 177.0504456

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3467741, upper bound: 195.3489035
time: 9.93 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3461601, upper bound: 195.3479629
time: 8.94 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -104.9692230, 83.1579132, -92.4082718, 73.1420441, -178.1112518, 175.5661774
1: -86.3696899, 72.5907288, -75.9085846, 63.7761002, -150.1457672, 148.4992981
2: -112.4341278, 71.6278458, -98.5751877, 62.5427284, -174.9768372, 170.2030029
3: -124.5501099, 62.0677834, -110.0107346, 54.3026886, -178.8527832, 172.0785217
4: -113.4068680, 86.2109833, -99.9523087, 75.8515472, -189.2584229, 186.1632690
5: -100.0609283, 78.1082764, -87.9972458, 68.5688782, -168.6297913, 166.1054993
6: -95.1717758, 93.1705627, -83.5991364, 82.1437836, -177.3155518, 176.7696533
7: -106.6246414, 83.4336395, -94.1258621, 72.6419296, -179.2665710, 177.5595093
8: -123.1846008, 84.3752060, -107.8815155, 73.8112030, -196.9958038, 192.2567139
9: -94.6163712, 93.7728882, -83.1070404, 82.4340668, -177.0504456, 176.8799286

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3485913, upper bound: 195.3506338
time: 11.12 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3479629, upper bound: 195.3496515
time: 11.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -104.9692230, 83.1579132, -104.9692230, 83.1579132, -188.1271362, 188.1271362
1: -86.3696899, 72.5907288, -86.3696899, 72.5907288, -158.9603577, 158.9603577
2: -112.4341278, 71.6278458, -112.4341278, 71.6278458, -184.0619202, 184.0619202
3: -124.5501099, 62.0677834, -124.5501099, 62.0677834, -186.6178741, 186.6178589
4: -113.4068680, 86.2109833, -113.4068680, 86.2109833, -199.6178436, 199.6178436
5: -100.0609283, 78.1082764, -100.0609283, 78.1082764, -178.1691895, 178.1691895
6: -95.1717758, 93.1705627, -95.1717758, 93.1705627, -188.3423157, 188.3423157
7: -106.6246414, 83.4336395, -106.6246414, 83.4336395, -190.0582886, 190.0582886
8: -123.1846008, 84.3752060, -123.1846008, 84.3752060, -207.5598145, 207.5598145
9: -94.6163712, 93.7728882, -94.6163712, 93.7728882, -188.3892517, 188.3892517

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3485913, upper bound: 195.3554506
time: 12.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3479629, upper bound: 195.3545877
time: 11.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.61 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.61
Output dim: 7, lower bound: -195.3467741, upper bound: 195.3472966
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.61
Output dim: 7, lower bound: -195.3461601, upper bound: 195.3461601
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.61
Output dim: 7, lower bound: -195.3467741, upper bound: 195.3489035
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.61
Output dim: 7, lower bound: -195.3461601, upper bound: 195.3479629
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.61
Output dim: 7, lower bound: -195.3485913, upper bound: 195.3506338
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.61
Output dim: 7, lower bound: -195.3479629, upper bound: 195.3496515
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.61
Output dim: 7, lower bound: -195.3485913, upper bound: 195.3554506
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.61
Output dim: 7, lower bound: -195.3479629, upper bound: 195.3545877

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -86.0807571, 68.1724548, -92.1563339, 72.9441071, -159.0248566, 160.3287811
1: -70.6390762, 59.2993851, -75.6986694, 63.5975990, -134.2366791, 134.9980164
2: -91.6216888, 58.0244179, -98.2981796, 62.3631744, -153.9848633, 156.3226013
3: -102.5155106, 50.4328957, -109.7121353, 54.1486588, -156.6641693, 160.1450043
4: -93.1055908, 70.6329498, -99.6797791, 75.6436920, -168.7492828, 170.3127136
5: -81.9639511, 63.7620430, -87.7570114, 68.3774872, -150.3414154, 151.5190582
6: -77.7842560, 76.5462723, -83.3677368, 81.9208832, -159.7051392, 159.9140015
7: -87.7513351, 67.2541885, -93.8721390, 72.4276505, -160.1789703, 161.1263275
8: -100.3265152, 68.6670303, -107.5807190, 73.6062393, -173.9327393, 176.2477417
9: -77.2512131, 76.7177887, -82.8738174, 82.2064972, -159.4577026, 159.5916138

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3379940, upper bound: 195.3380831
time: 8.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3379225, upper bound: 195.3380973
time: 8.19 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -87.9659805, 69.6393585, -91.6852417, 72.5731812, -160.5391541, 161.3246002
1: -72.1701050, 60.5365105, -75.3054276, 63.2623291, -135.4324036, 135.8419037
2: -93.5136566, 59.0862846, -97.7767410, 62.0221405, -155.5357819, 156.8630066
3: -104.8251038, 51.3987885, -109.1549911, 53.8568420, -158.6819305, 160.5537720
4: -95.1564713, 72.1181030, -99.1692657, 75.2537155, -170.4101868, 171.2873688
5: -83.6750107, 65.0227356, -87.3051834, 68.0162201, -151.6912231, 152.3278809
6: -79.3977814, 78.2398529, -82.9314575, 81.5035019, -160.9012756, 161.1713104
7: -89.6711426, 68.4204559, -93.3962250, 72.0197067, -161.6908569, 161.8166809
8: -102.4377365, 70.1077194, -107.0159912, 73.2219467, -175.6596832, 177.1237183
9: -78.8514633, 78.3254471, -82.4352341, 81.7785416, -160.6299896, 160.7606659

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3372383, upper bound: 195.3372176
time: 9.09 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3372358, upper bound: 195.3372358
time: 9.62 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -86.0807571, 68.1724548, -104.7016525, 82.9472046, -169.0279388, 172.8741150
1: -70.6390762, 59.2993851, -86.1445999, 72.4006195, -143.0397034, 145.4439545
2: -91.6216888, 58.0244179, -112.1390305, 71.4377060, -163.0593872, 170.1634521
3: -102.5155106, 50.4328957, -124.2334824, 61.9033012, -164.4188080, 174.6663513
4: -93.1055908, 70.6329498, -113.1173172, 85.9889221, -179.0945129, 183.7502594
5: -81.9639511, 63.7620430, -99.8048859, 77.9054718, -159.8694153, 163.5669250
6: -77.7842560, 76.5462723, -94.9261246, 92.9330750, -170.7173309, 171.4723663
7: -87.7513351, 67.2541885, -106.3546371, 83.2067261, -170.9580688, 173.6088257
8: -100.3265152, 68.6670303, -122.8645096, 84.1569595, -184.4834595, 191.5315247
9: -77.2512131, 76.7177887, -94.3694763, 93.5303497, -170.7815552, 171.0872650

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3420650, upper bound: 195.3397827
time: 11.19 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3379225, upper bound: 195.3398633
time: 9.82 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -87.9659805, 69.6393585, -104.2358780, 82.5800705, -170.5460358, 173.8752441
1: -72.1701050, 60.5365105, -85.7553101, 72.0695267, -144.2396240, 146.2918091
2: -93.5136566, 59.0862846, -111.6227112, 71.1030197, -164.6166687, 170.7089996
3: -104.8251038, 51.3987885, -123.6833267, 61.6143913, -166.4394836, 175.0821228
4: -95.1564713, 72.1181030, -112.6124039, 85.6013641, -180.7578430, 184.7304993
5: -83.6750107, 65.0227356, -99.3571701, 77.5494919, -161.2244873, 164.3798676
6: -79.3977814, 78.2398529, -94.4955597, 92.5196075, -171.9173737, 172.7354126
7: -89.6711426, 68.4204559, -105.8840561, 82.8061676, -172.4772949, 174.3044586
8: -102.4377365, 70.1077194, -122.3047867, 83.7769699, -186.2147064, 192.4125061
9: -78.8514633, 78.3254471, -93.9377060, 93.1078186, -171.9592285, 172.2631378

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3415896, upper bound: 195.3390366
time: 12.75 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3415712, upper bound: 195.3390093
time: 13.02 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -98.2581329, 77.8778534, -92.1563339, 72.9441071, -171.2022400, 170.0341797
1: -80.7665787, 67.8431778, -75.6986694, 63.5975990, -144.3641815, 143.5418396
2: -105.0398788, 66.8529282, -98.2981796, 62.3631744, -167.4030457, 165.1511078
3: -116.6141815, 57.9503212, -109.7121353, 54.1486588, -170.7628021, 167.6624451
4: -106.1369934, 80.6508102, -99.6797791, 75.6436920, -181.7806702, 180.3305969
5: -93.6411743, 73.0181885, -87.7570114, 68.3774872, -162.0186615, 160.7751770
6: -89.0047150, 87.2230759, -83.3677368, 81.9208832, -170.9255829, 170.5908203
7: -99.8584290, 77.7346115, -93.8721390, 72.4276505, -172.2860718, 171.6067505
8: -115.1531677, 78.9095306, -107.5807190, 73.6062393, -188.7593994, 186.4902496
9: -88.4194183, 87.7053452, -82.8738174, 82.2064972, -170.6259155, 170.5791626

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3396765, upper bound: 195.3424337
time: 11.14 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3396060, upper bound: 195.3424796
time: 14.15 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -100.3584137, 79.5168610, -91.6852417, 72.5731812, -172.9315796, 171.2021027
1: -82.4760895, 69.2385178, -75.3054276, 63.2623291, -145.7384033, 144.5439453
2: -107.1818695, 68.0916748, -97.7767410, 62.0221405, -169.2040100, 165.8683929
3: -119.1716614, 59.0648499, -109.1549911, 53.8568420, -173.0285034, 168.2198181
4: -108.4165573, 82.3142853, -99.1692657, 75.2537155, -183.6702728, 181.4835510
5: -95.5569153, 74.4586639, -87.3051834, 68.0162201, -163.5731354, 161.7638397
6: -90.8289948, 89.1049042, -82.9314575, 81.5035019, -172.3324738, 172.0363617
7: -101.9972534, 79.1226959, -93.3962250, 72.0197067, -174.0169678, 172.5189056
8: -117.5351562, 80.5285339, -107.0159912, 73.2219467, -190.7570801, 187.5445251
9: -90.2338104, 89.5166550, -82.4352341, 81.7785416, -172.0123596, 171.9518890

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3390790, upper bound: 195.3415683
time: 12.11 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3390093, upper bound: 195.3415712
time: 8.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -98.2581329, 77.8778534, -104.7016525, 82.9472046, -181.2053375, 182.5794983
1: -80.7665787, 67.8431778, -86.1445999, 72.4006195, -153.1672058, 153.9877777
2: -105.0398788, 66.8529282, -112.1390305, 71.4377060, -176.4775696, 178.9919586
3: -116.6141815, 57.9503212, -124.2334824, 61.9033012, -178.5174713, 182.1837769
4: -106.1369934, 80.6508102, -113.1173172, 85.9889221, -192.1259155, 193.7681274
5: -93.6411743, 73.0181885, -99.8048859, 77.9054718, -171.5466461, 172.8230286
6: -89.0047150, 87.2230759, -94.9261246, 92.9330750, -181.9377899, 182.1491852
7: -99.8584290, 77.7346115, -106.3546371, 83.2067261, -183.0651550, 184.0892487
8: -115.1531677, 78.9095306, -122.8645096, 84.1569595, -199.3100891, 201.7740479
9: -88.4194183, 87.7053452, -94.3694763, 93.5303497, -181.9497681, 182.0748291

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3474187, upper bound: 195.3476907
time: 11.07 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3472142, upper bound: 195.3475798
time: 11.63 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -100.3584137, 79.5168610, -104.2358780, 82.5800705, -182.9384766, 183.7527313
1: -82.4760895, 69.2385178, -85.7553101, 72.0695267, -154.5456238, 154.9938354
2: -107.1818695, 68.0916748, -111.6227112, 71.1030197, -178.2848816, 179.7143860
3: -119.1716614, 59.0648499, -123.6833267, 61.6143913, -180.7860565, 182.7481689
4: -108.4165573, 82.3142853, -112.6124039, 85.6013641, -194.0179138, 194.9266663
5: -95.5569153, 74.4586639, -99.3571701, 77.5494919, -173.1063843, 173.8158264
6: -90.8289948, 89.1049042, -94.4955597, 92.5196075, -183.3485565, 183.6004639
7: -101.9972534, 79.1226959, -105.8840561, 82.8061676, -184.8034210, 185.0066681
8: -117.5351562, 80.5285339, -122.3047867, 83.7769699, -201.3121338, 202.8333130
9: -90.2338104, 89.5166550, -93.9377060, 93.1078186, -183.3416138, 183.4543457

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3469422, upper bound: 195.3467763
time: 9.89 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3466678, upper bound: 195.3466680
time: 10.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.38 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3379940, upper bound: 195.3380831
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3379225, upper bound: 195.3380973
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3372383, upper bound: 195.3372176
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3372358, upper bound: 195.3372358
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3420650, upper bound: 195.3397827
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3379225, upper bound: 195.3398633
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3415896, upper bound: 195.3390366
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3415712, upper bound: 195.3390093
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3396765, upper bound: 195.3424337
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3396060, upper bound: 195.3424796
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3390790, upper bound: 195.3415683
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3390093, upper bound: 195.3415712
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3474187, upper bound: 195.3476907
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3472142, upper bound: 195.3475798
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3469422, upper bound: 195.3467763
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.38
Output dim: 7, lower bound: -195.3466678, upper bound: 195.3466680

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -85.8340683, 67.9777374, -90.2441025, 71.5263290, -157.3603821, 158.2218323
1: -70.4357452, 59.1271667, -74.1895828, 62.3081970, -132.7439423, 133.3167267
2: -91.3513870, 57.8496933, -96.2866821, 61.2489967, -152.6003723, 154.1363831
3: -102.2270126, 50.2827225, -107.3387070, 53.0849075, -155.3119202, 157.6214294
4: -92.8417969, 70.4309769, -97.6507111, 74.1254578, -166.9672241, 168.0816956
5: -81.7290192, 63.5765610, -85.9914093, 67.0658035, -148.7948303, 149.5679626
6: -77.5572815, 76.3297882, -81.6461182, 80.2280960, -157.7853699, 157.9759064
7: -87.5055389, 67.0461502, -91.9548569, 71.0782013, -158.5837250, 159.0010071
8: -100.0297089, 68.4635391, -105.4885025, 72.2349854, -172.2646942, 173.9520264
9: -77.0259552, 76.4970398, -81.2099380, 80.5988464, -157.6248016, 157.7069702

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3375440, upper bound: 195.3393519
time: 10.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3375440, upper bound: 195.3396038
time: 11.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -83.9847488, 66.5197906, -93.3127441, 73.9307327, -157.9154358, 159.8325348
1: -68.9127121, 57.8374214, -76.7035065, 64.3959274, -133.3086395, 134.5409241
2: -89.3277512, 56.5398064, -99.5558014, 63.2593842, -152.5871277, 156.0956116
3: -100.0613861, 49.1555290, -111.0025177, 54.8308907, -154.8922577, 160.1580505
4: -90.8606796, 68.9212418, -100.9502029, 76.6340561, -167.4947357, 169.8714294
5: -79.9723358, 62.1891479, -88.9038849, 69.3141098, -149.2864380, 151.0930328
6: -75.8592148, 74.7058792, -84.4085846, 82.9455643, -158.8047791, 159.1144714
7: -85.6611176, 65.4935532, -95.0673141, 73.4464951, -159.1076050, 160.5608673
8: -97.8082886, 66.9390182, -109.0551071, 74.6226273, -172.4309082, 175.9941254
9: -75.3384628, 74.8457260, -83.9724197, 83.3110809, -158.6495361, 158.8181458

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3306465, upper bound: 195.3284609
time: 12.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3238076, upper bound: 195.3198869
time: 10.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -87.7198334, 69.4451447, -89.7920761, 71.1715622, -158.8913879, 159.2372131
1: -71.9676056, 60.3647385, -73.8131485, 61.9873466, -133.9549408, 134.1778717
2: -93.2439270, 58.9119339, -95.7873764, 60.9224052, -154.1663361, 154.6992798
3: -104.5377121, 51.2489738, -106.8053513, 52.8048630, -157.3425598, 158.0543213
4: -94.8935547, 71.9166718, -97.1605682, 73.7514267, -168.6449890, 169.0772247
5: -83.4407120, 64.8375854, -85.5583649, 66.7195358, -150.1602478, 150.3959503
6: -79.1714096, 78.0241470, -81.2278442, 79.8279953, -158.9993896, 159.2519836
7: -89.4259491, 68.2127914, -91.4987946, 70.6880264, -160.1139526, 159.7115784
8: -102.1414490, 69.9047241, -104.9477081, 71.8666382, -174.0080566, 174.8524017
9: -78.6267471, 78.1053467, -80.7900543, 80.1887589, -158.8155060, 158.8954010

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3411617, upper bound: 195.3385885
time: 10.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3411617, upper bound: 195.3388356
time: 9.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -85.8999939, 68.0092392, -92.8685760, 73.5818405, -159.4818420, 160.8778076
1: -70.4702835, 59.0957375, -76.3328400, 64.0806122, -134.5509033, 135.4285736
2: -91.2511444, 57.6221962, -99.0644760, 62.9385033, -154.1896057, 156.6866760
3: -102.4083633, 50.1397400, -110.4780121, 54.5555077, -156.9638672, 160.6177521
4: -92.9451904, 70.4304657, -100.4687042, 76.2662201, -169.2113953, 170.8991699
5: -81.7113037, 63.4714737, -88.4776840, 68.9739761, -150.6852722, 151.9491577
6: -77.5002518, 76.4271698, -83.9977646, 82.5523148, -160.0525665, 160.4249115
7: -87.6116867, 66.6830292, -94.6196213, 73.0627899, -160.6744690, 161.3026123
8: -99.9525146, 68.4034271, -108.5234604, 74.2602234, -174.2127380, 176.9268799
9: -76.9667892, 76.4800949, -83.5597153, 82.9078293, -159.8746185, 160.0398102

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3301788, upper bound: 195.3277024
time: 12.29 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3234323, upper bound: 195.3193190
time: 11.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -98.0052032, 77.6779099, -78.2149582, 61.9572906, -159.9624939, 155.8928680
1: -80.5573120, 67.6666412, -64.2112885, 53.8763962, -134.4337006, 131.8779297
2: -104.7623444, 66.6744995, -83.0436935, 52.5059280, -157.2682800, 149.7181854
3: -116.3188477, 57.7962341, -93.4115677, 45.6627121, -161.9815521, 151.2077789
4: -105.8664551, 80.4433289, -84.7810440, 64.2380371, -170.1044922, 165.2243652
5: -93.3994522, 72.8284225, -74.4823151, 57.9203873, -151.3198395, 147.3107300
6: -88.7725067, 87.0007858, -70.5563507, 69.6925583, -158.4650574, 157.5571289
7: -99.6067886, 77.5221481, -79.9805756, 60.6934776, -160.3002625, 157.5027008
8: -114.8494797, 78.7005844, -90.8570023, 62.1472893, -176.9967651, 169.5575714
9: -88.1890640, 87.4790268, -70.1545410, 69.7553635, -157.9444275, 157.6335602

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3392453, upper bound: 195.3420690
time: 9.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3392453, upper bound: 195.3422603
time: 14.26 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -96.0864258, 76.1630402, -81.3274841, 64.3961639, -160.4825592, 157.4905090
1: -78.9705353, 66.3264618, -66.7631531, 55.9964790, -134.9670105, 133.0896149
2: -102.6597137, 65.3194809, -86.3635864, 54.5423431, -157.2020264, 151.6830444
3: -114.0763245, 56.6238708, -97.1301422, 47.4317780, -161.5081024, 153.7539978
4: -103.8084106, 78.8724213, -88.1368484, 66.7891464, -170.5975342, 167.0092468
5: -91.5692596, 71.3914032, -77.4362335, 60.1955032, -151.7647552, 148.8276367
6: -87.0123444, 85.3132706, -73.3547363, 72.4550400, -159.4673767, 158.6679993
7: -97.6946030, 75.9156647, -83.1464310, 63.0886650, -160.7832642, 159.0621033
8: -112.5453110, 77.1155853, -94.4730301, 64.5740280, -177.1193390, 171.5886078
9: -86.4418182, 85.7629852, -72.9562073, 72.5129776, -158.9547882, 158.7191925

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3212406, upper bound: 195.3255230
time: 12.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3198543, upper bound: 195.3242705
time: 11.15 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -100.1060944, 79.3173904, -77.7665710, 61.6054382, -161.7115326, 157.0839539
1: -82.2674561, 69.0623169, -63.8396721, 53.5592575, -135.8267059, 132.9019470
2: -106.9049683, 67.9135208, -82.5482941, 52.1807060, -159.0856781, 150.4618225
3: -118.8771362, 58.9110565, -92.8821259, 45.3854446, -164.2625580, 151.7931824
4: -108.1468048, 82.1073074, -84.2957764, 63.8664894, -172.0132904, 166.4030304
5: -95.3157959, 74.2693100, -74.0531693, 57.5767593, -152.8925476, 148.3224792
6: -90.5973053, 88.8832245, -70.1406021, 69.2965851, -159.8938904, 159.0238190
7: -101.7462387, 78.9104309, -79.5266876, 60.3047562, -162.0509949, 158.4371185
8: -117.2321014, 80.3200836, -90.3213959, 61.7836075, -179.0156860, 170.6414337
9: -90.0038300, 89.2909012, -69.7367249, 69.3487015, -159.3524628, 159.0276031

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3385885, upper bound: 195.3411617
time: 12.07 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3385885, upper bound: 195.3413842
time: 11.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -98.2051849, 77.8160706, -80.8877869, 64.0509644, -162.2561493, 158.7038422
1: -80.6953812, 67.7347488, -66.3984299, 55.6855850, -136.3809662, 134.1331635
2: -104.8214874, 66.5699844, -85.8775330, 54.2234879, -159.0449524, 152.4475098
3: -116.6559067, 57.7490234, -96.6108475, 47.1599426, -163.8158569, 154.3598480
4: -106.1091309, 80.5505447, -87.6612244, 66.4248505, -172.5339813, 168.2117615
5: -93.5020981, 72.8445053, -77.0150452, 59.8587952, -153.3608704, 149.8595428
6: -88.8537750, 87.2116241, -72.9471130, 72.0667419, -160.9205170, 160.1587372
7: -99.8526077, 77.3169785, -82.7016220, 62.7073936, -162.5599670, 160.0185547
8: -114.9488449, 78.7488251, -93.9474106, 64.2175903, -179.1664429, 172.6962280
9: -88.2727432, 87.5907593, -72.5469284, 72.1138000, -160.3865356, 160.1376801

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3206604, upper bound: 195.3247284
time: 10.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3193190, upper bound: 195.3234323
time: 11.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -98.0052032, 77.6779099, -90.2441025, 71.5263290, -169.5315247, 167.9220123
1: -80.5573120, 67.6666412, -74.1895828, 62.3081970, -142.8654785, 141.8562012
2: -104.7623444, 66.6744995, -96.2866821, 61.2489967, -166.0113068, 162.9611816
3: -116.3188477, 57.7962341, -107.3387070, 53.0849075, -169.4037476, 165.1349487
4: -105.8664551, 80.4433289, -97.6507111, 74.1254578, -179.9919128, 178.0940399
5: -93.3994522, 72.8284225, -85.9914093, 67.0658035, -160.4652557, 158.8198242
6: -88.7725067, 87.0007858, -81.6461182, 80.2280960, -169.0006104, 168.6469116
7: -99.6067886, 77.5221481, -91.9548569, 71.0782013, -170.6849823, 169.4770050
8: -114.8494797, 78.7005844, -105.4885025, 72.2349854, -187.0844727, 184.1890717
9: -88.1890640, 87.4790268, -81.2099380, 80.5988464, -168.7879028, 168.6889648

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3469758, upper bound: 195.3473443
time: 10.27 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3469758, upper bound: 195.3474363
time: 9.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -96.0864258, 76.1630402, -93.3127441, 73.9307327, -170.0171509, 169.4757690
1: -78.9705353, 66.3264618, -76.7035065, 64.3959274, -143.3664551, 143.0299683
2: -102.6597137, 65.3194809, -99.5558014, 63.2593842, -165.9190979, 164.8752594
3: -114.0763245, 56.6238708, -111.0025177, 54.8308907, -168.9072113, 167.6263733
4: -103.8084106, 78.8724213, -100.9502029, 76.6340561, -180.4424591, 179.8225861
5: -91.5692596, 71.3914032, -88.9038849, 69.3141098, -160.8833618, 160.2952881
6: -87.0123444, 85.3132706, -84.4085846, 82.9455643, -169.9579163, 169.7218628
7: -97.6946030, 75.9156647, -95.0673141, 73.4464951, -171.1410980, 170.9829712
8: -112.5453110, 77.1155853, -109.0551071, 74.6226273, -187.1679382, 186.1706848
9: -86.4418182, 85.7629852, -83.9724197, 83.3110809, -169.7528839, 169.7354126

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3377101, upper bound: 195.3383996
time: 12.08 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3363303, upper bound: 195.3366501
time: 11.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.95 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3375440, upper bound: 195.3393519
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3375440, upper bound: 195.3396038
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3306465, upper bound: 195.3284609
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3238076, upper bound: 195.3198869
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3411617, upper bound: 195.3385885
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3411617, upper bound: 195.3388356
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3301788, upper bound: 195.3277024
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3234323, upper bound: 195.3193190
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3392453, upper bound: 195.3420690
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3392453, upper bound: 195.3422603
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3212406, upper bound: 195.3255230
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3198543, upper bound: 195.3242705
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3385885, upper bound: 195.3411617
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3385885, upper bound: 195.3413842
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3206604, upper bound: 195.3247284
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3193190, upper bound: 195.3234323
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3469758, upper bound: 195.3473443
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3469758, upper bound: 195.3474363
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3377101, upper bound: 195.3383996
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.95
Output dim: 7, lower bound: -195.3363303, upper bound: 195.3366501
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.95
Output dim: 7, lower bound: -195.3469422, upper bound: 195.3467763
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.95
Output dim: 7, lower bound: -195.3466678, upper bound: 195.3466680
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.82974243164062
rel_dist={7: [-195.364420828844, 195.364420828844]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3537704, upper bound: 195.3527759
time: 14.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3599442, upper bound: 195.3599442
time: 12.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 27.08 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 27.08
Output dim: 7, lower bound: -195.3537704, upper bound: 195.3527759
IS_A2, status: Status.UNKNOWN, split count: 1, time: 27.08
Output dim: 7, lower bound: -195.3599442, upper bound: 195.3599442

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -92.4082718, 73.1420441, -105.0767975, 83.2133102, -175.6215668, 178.2188416
1: -75.9085846, 63.7761002, -86.4515839, 72.6697083, -148.5782928, 150.2276611
2: -98.5751877, 62.5427284, -112.5323868, 71.6675415, -170.2427216, 175.0751190
3: -110.0107346, 54.3026886, -124.7410202, 62.1198082, -172.1305389, 179.0436707
4: -99.9523087, 75.8515472, -113.5291367, 86.2810440, -186.2333527, 189.3806763
5: -87.9972458, 68.5688782, -100.1505432, 78.1624298, -166.1596375, 168.7194214
6: -83.5991364, 82.1437836, -95.2577057, 93.2733459, -176.8724670, 177.4014740
7: -94.1258621, 72.6419296, -106.7559891, 83.4716949, -177.5975342, 179.3979034
8: -107.8815155, 73.8112030, -123.2623596, 84.3909454, -192.2724304, 197.0735626
9: -83.1070404, 82.4340668, -94.7092972, 93.8457184, -176.9527588, 177.1433563

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3488248, upper bound: 195.3474564
time: 14.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3483109, upper bound: 195.3471664
time: 15.16 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -104.9692230, 83.1579132, -111.1472092, 88.0602417, -193.0294647, 194.3051147
1: -86.3696899, 72.5907288, -91.5578461, 76.9445038, -163.3141785, 164.1485291
2: -112.4341278, 71.6278458, -119.2405472, 76.0385895, -188.4727020, 190.8683624
3: -124.5501099, 62.0677834, -131.7770691, 65.8847809, -190.4348755, 193.8448181
4: -113.4068680, 86.2109833, -120.0414886, 91.3031540, -204.7100220, 206.2524567
5: -100.0609283, 78.1082764, -105.9926071, 82.7648087, -182.8256989, 184.1008911
6: -95.1717758, 93.1705627, -100.8442535, 98.6188278, -193.7906036, 194.0148163
7: -106.6246414, 83.4336395, -112.8124619, 88.6633911, -195.2880096, 196.2460785
8: -123.1846008, 84.3752060, -130.6515350, 89.4954681, -212.6800690, 215.0267334
9: -94.6163712, 93.7728882, -100.2624893, 99.3480530, -193.9644165, 194.0353546

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3550219, upper bound: 195.3547659
time: 10.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3544746, upper bound: 195.3544746
time: 11.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.87 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.87
Output dim: 7, lower bound: -195.3488248, upper bound: 195.3474564
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.87
Output dim: 7, lower bound: -195.3483109, upper bound: 195.3471664
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.87
Output dim: 7, lower bound: -195.3550219, upper bound: 195.3547659
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.87
Output dim: 7, lower bound: -195.3544746, upper bound: 195.3544746

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -89.9373016, 71.2017059, -98.4789124, 78.0244141, -167.9617157, 169.6806030
1: -73.8503036, 62.0264854, -80.9455490, 68.0057220, -141.8560181, 142.9720001
2: -95.8591614, 60.7803993, -105.2661667, 66.9801712, -162.8392944, 166.0465546
3: -107.0830078, 52.7922935, -116.9376907, 58.0752716, -165.1582794, 169.7299805
4: -97.2794571, 73.8136520, -106.3832245, 80.8147736, -178.0942383, 180.1968689
5: -85.6413498, 66.6919556, -93.8425140, 73.1616592, -158.8030090, 160.5344696
6: -81.3298340, 79.9580765, -89.1986542, 87.4281006, -168.7579346, 169.1567383
7: -91.6371307, 70.5393219, -100.1050720, 77.8760834, -169.5131989, 170.6443787
8: -104.9316711, 71.8024750, -115.3676376, 79.0207062, -183.9523773, 187.1701050
9: -80.8204727, 80.2024994, -88.6218109, 87.8847427, -168.7051849, 168.8243103

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3400583, upper bound: 195.3382851
time: 15.04 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3403253, upper bound: 195.3384226
time: 13.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -89.0694885, 70.5168686, -100.3997803, 79.5184402, -168.5879211, 170.9166412
1: -73.1259918, 61.4055939, -82.5035095, 69.2685394, -142.3945312, 143.9091034
2: -94.8898468, 60.1353188, -107.2036285, 68.0804138, -162.9702301, 167.3389282
3: -106.0609818, 52.2425270, -119.2833481, 59.0720177, -165.1329956, 171.5258636
4: -96.3367386, 73.0916748, -108.4662247, 82.3242950, -178.6610107, 181.5578918
5: -84.8029633, 66.0154648, -95.5815887, 74.4591904, -159.2621307, 161.5970001
6: -80.5155106, 79.1884995, -90.8499222, 89.1492233, -169.6647339, 170.0384064
7: -90.7570038, 69.7658539, -102.0561295, 79.0955505, -169.8525543, 171.8219910
8: -103.8845901, 71.0915756, -117.5270844, 80.4910507, -184.3756256, 188.6186523
9: -80.0051270, 79.4073868, -90.2612000, 89.5263062, -169.5314331, 169.6685791

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3396318, upper bound: 195.3379632
time: 15.89 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3398579, upper bound: 195.3380849
time: 15.86 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -102.3438644, 81.0916519, -104.3980789, 82.7384491, -185.0823059, 185.4897308
1: -84.1764908, 70.7321167, -85.8754883, 72.1503372, -156.3267975, 156.6075897
2: -109.5410004, 69.7625198, -111.7947235, 71.2447815, -180.7857513, 181.5572510
3: -121.4430542, 60.4572830, -123.7949142, 61.7359009, -183.1789551, 184.2521820
4: -110.5636292, 84.0335693, -112.7408981, 85.7017746, -196.2654114, 196.7744446
5: -97.5505371, 76.1170425, -99.5291977, 77.6495514, -175.2000885, 175.6462402
6: -92.7591171, 90.8430176, -94.6461945, 92.6282578, -185.3873749, 185.4891968
7: -103.9757233, 81.2069092, -106.0011749, 82.9422607, -186.9179840, 187.2080688
8: -120.0408478, 82.2361984, -122.5717239, 83.9945679, -204.0353851, 204.8079224
9: -92.1927261, 91.3992691, -94.0344620, 93.2287445, -185.4214478, 185.4337311

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3470948, upper bound: 195.3468678
time: 12.67 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3471445, upper bound: 195.3468473
time: 14.25 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -101.5690231, 80.4797974, -106.3823395, 84.2860641, -185.8550873, 186.8621216
1: -83.5271606, 70.1773529, -87.4868774, 73.4625320, -156.9896851, 157.6642303
2: -108.6734467, 69.1929855, -113.8076019, 72.4030380, -181.0764771, 183.0005493
3: -120.5314331, 59.9667778, -126.2121964, 62.7788010, -183.3102417, 186.1789703
4: -109.7217255, 83.3858948, -114.8897324, 87.2649841, -196.9867096, 198.2756042
5: -96.7992325, 75.5169830, -101.3328705, 79.0021515, -175.8013763, 176.8498535
6: -92.0353394, 90.1539536, -96.3626709, 94.4051590, -186.4404907, 186.5166016
7: -103.1912994, 80.5233612, -108.0170288, 84.2328262, -187.4241028, 188.5403900
8: -119.1037140, 81.6023407, -124.8116150, 85.5195312, -204.6232452, 206.4139557
9: -91.4690475, 90.6914062, -95.7404327, 94.9335938, -186.4026489, 186.4318085

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3465015, upper bound: 195.3466107
time: 11.69 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3465444, upper bound: 195.3465444
time: 11.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.41
Output dim: 7, lower bound: -195.3400583, upper bound: 195.3382851
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.41
Output dim: 7, lower bound: -195.3403253, upper bound: 195.3384226
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.41
Output dim: 7, lower bound: -195.3396318, upper bound: 195.3379632
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.41
Output dim: 7, lower bound: -195.3398579, upper bound: 195.3380849
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.41
Output dim: 7, lower bound: -195.3470948, upper bound: 195.3468678
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.41
Output dim: 7, lower bound: -195.3471445, upper bound: 195.3468473
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.41
Output dim: 7, lower bound: -195.3465015, upper bound: 195.3466107
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.41
Output dim: 7, lower bound: -195.3465444, upper bound: 195.3465444

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -76.1262817, 60.3207207, -92.8627777, 73.5848389, -149.7111206, 153.1835022
1: -62.4809036, 52.4075470, -76.3012238, 64.0792007, -126.5600815, 128.7087708
2: -80.7526398, 51.0177002, -99.1058121, 63.0151749, -143.7678070, 150.1235046
3: -90.9368210, 44.3906403, -110.3738403, 54.6530609, -145.5898438, 154.7644501
4: -82.5218124, 62.5129662, -100.3793182, 76.2093353, -158.7311401, 162.8922577
5: -72.4930649, 56.3360405, -88.4765701, 68.9472809, -141.4403381, 144.8125916
6: -68.6372604, 67.8515320, -84.0399628, 82.4957733, -151.1330261, 151.8914948
7: -77.8705215, 58.9168777, -94.5132675, 73.1534119, -151.0239258, 153.4300842
8: -88.3720703, 60.4586258, -108.6195297, 74.3861389, -162.7582092, 169.0781403
9: -68.2203598, 67.8710861, -83.5023575, 82.8589783, -151.0793152, 151.3734436

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3205815, upper bound: 195.3179271
time: 12.61 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3199324, upper bound: 195.3172984
time: 13.53 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -79.1967621, 62.7276917, -91.6557999, 72.6343918, -151.8311462, 154.3834839
1: -64.9981308, 54.4969788, -75.3030243, 63.2352295, -128.2333679, 129.7999878
2: -84.0237732, 53.0210152, -97.7905426, 62.1610146, -146.1847839, 150.8115540
3: -94.6092072, 46.1321754, -108.9533234, 53.9136314, -148.5228271, 155.0854797
4: -85.8353653, 65.0304337, -99.0754242, 75.2325668, -161.0679169, 164.1058655
5: -75.4065704, 58.5799255, -87.3355408, 68.0483551, -143.4549103, 145.9154663
6: -71.3951263, 70.5756073, -82.9415359, 81.4294510, -152.8245544, 153.5171356
7: -80.9953690, 61.2727585, -93.3069992, 72.1578064, -153.1531677, 154.5797577
8: -91.9366913, 62.8519249, -107.1715317, 73.3872070, -165.3238678, 170.0234528
9: -70.9840164, 70.5916595, -82.4048920, 81.7872849, -152.7713013, 152.9965515

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3210806, upper bound: 195.3183218
time: 12.85 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3204498, upper bound: 195.3176326
time: 13.45 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -75.2987823, 59.6717567, -94.7731934, 75.0715027, -150.3702850, 154.4449310
1: -61.7975006, 51.8191948, -77.8533096, 65.3342972, -127.1317825, 129.6724854
2: -79.8306122, 50.4038849, -101.0309372, 64.1055984, -143.9362183, 151.4348145
3: -89.9655533, 43.8672714, -112.7094879, 55.6427155, -145.6082611, 156.5767517
4: -81.6244888, 61.8261070, -102.4541626, 77.7128601, -159.3373260, 164.2802734
5: -71.6996765, 55.6882210, -90.2087250, 70.2330475, -141.9326935, 145.8969421
6: -67.8631058, 67.1243820, -85.6821289, 84.2085648, -152.0716705, 152.8065186
7: -77.0301208, 58.1777000, -96.4559174, 74.3588104, -151.3889313, 154.6335602
8: -87.3793030, 59.7838097, -110.7664032, 75.8458939, -163.2251892, 170.5502167
9: -67.4406433, 67.1148605, -85.1286621, 84.4923172, -151.9329529, 152.2435303

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3201766, upper bound: 195.3176522
time: 12.45 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3195596, upper bound: 195.3170439
time: 13.00 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -78.4349747, 62.1273041, -93.6242981, 74.1663666, -152.6013489, 155.7516022
1: -64.3665543, 53.9540024, -76.9012985, 64.5324554, -128.8989868, 130.8553009
2: -83.1715851, 52.4521408, -99.7793427, 63.2922401, -146.4638062, 152.2314758
3: -93.7141266, 45.6485748, -111.3559875, 54.9386940, -148.6528168, 157.0045624
4: -85.0091782, 64.3960800, -101.2130661, 76.7819443, -161.7911224, 165.6091309
5: -74.6714935, 57.9828415, -89.1227798, 69.3777466, -144.0492401, 147.1056213
6: -70.6799088, 69.9051895, -84.6378632, 83.1944885, -153.8743896, 154.5430298
7: -80.2224884, 60.5870667, -95.3088913, 73.4107819, -153.6332550, 155.8959656
8: -91.0184326, 62.2291031, -109.3873062, 74.8947830, -165.9131927, 171.6163940
9: -70.2658997, 69.8924179, -84.0849991, 83.4714966, -153.7373810, 153.9773712

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3206493, upper bound: 195.3180246
time: 12.33 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3200305, upper bound: 195.3173514
time: 16.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -87.9768066, 69.7466125, -98.7131958, 78.2421646, -166.2189331, 168.4597473
1: -72.3011932, 60.7051430, -81.1706467, 68.1801682, -140.4813538, 141.8757935
2: -93.7948227, 59.6295967, -105.5529633, 67.2410889, -161.0359039, 165.1825562
3: -104.6539917, 51.6955948, -117.1539993, 58.2685928, -162.9225769, 168.8495941
4: -95.1933899, 72.2532959, -106.6585083, 81.0295715, -176.2229614, 178.9118042
5: -83.8285599, 65.3426514, -94.0946732, 73.3851013, -157.2136078, 159.4373169
6: -79.5602264, 78.2224045, -89.4266968, 87.6289749, -167.1891785, 167.6491089
7: -89.6692657, 69.1474915, -100.3381805, 78.1712112, -167.8404846, 169.4856720
8: -102.7812958, 70.3907166, -115.7276917, 79.3053360, -182.0866089, 186.1183929
9: -79.1116867, 78.5481567, -88.8616714, 88.1425247, -167.2542114, 167.4098206

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3369541, upper bound: 195.3366497
time: 15.13 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3359791, upper bound: 195.3358668
time: 16.52 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -91.0390244, 72.1455994, -97.4781342, 77.2731094, -168.3121338, 169.6237335
1: -74.8094788, 62.7874336, -80.1473465, 67.3182755, -142.1277466, 142.9347839
2: -97.0546951, 61.6334915, -104.2056503, 66.3678360, -163.4225159, 165.8391266
3: -108.3122253, 53.4373207, -115.7022476, 57.5100021, -165.8221893, 169.1395721
4: -98.4882965, 74.7573471, -105.3192062, 80.0269089, -178.5151978, 180.0765533
5: -86.7346725, 67.5856476, -92.9260254, 72.4680634, -159.2027283, 160.5116730
6: -82.3162308, 80.9345322, -88.3016434, 86.5384903, -168.8547211, 169.2361298
7: -92.7769470, 71.5061417, -99.1020355, 77.1547394, -169.9316864, 170.6081848
8: -106.3384781, 72.7721481, -114.2477570, 78.2824936, -184.6209564, 187.0198975
9: -81.8672943, 81.2546616, -87.7409363, 87.0437775, -168.9110718, 168.9955750

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3373022, upper bound: 195.3368271
time: 11.67 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3362465, upper bound: 195.3360153
time: 11.92 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -87.2238541, 69.1540222, -100.6932526, 79.7863922, -167.0102386, 169.8472748
1: -71.6733170, 60.1654816, -82.7795334, 69.4879379, -141.1612549, 142.9450073
2: -92.9547424, 59.0723991, -107.5600967, 68.3922882, -161.3470154, 166.6324921
3: -103.7687149, 51.2196732, -119.5691605, 59.3076630, -163.0763702, 170.7888336
4: -94.3764038, 71.6277695, -108.8052979, 82.5908356, -176.9672394, 180.4330444
5: -83.1017761, 64.7571335, -95.8946457, 74.7326508, -157.8344269, 160.6517792
6: -78.8561096, 77.5546112, -91.1378860, 89.4031830, -168.2592926, 168.6925049
7: -88.9078979, 68.4793396, -102.3522034, 79.4534760, -168.3613739, 170.8315430
8: -101.8740234, 69.7754974, -117.9654694, 80.8242569, -182.6982422, 187.7409668
9: -78.4051361, 77.8604736, -90.5596619, 89.8439255, -168.2490234, 168.4201355

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3363707, upper bound: 195.3363519
time: 13.54 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3354469, upper bound: 195.3355856
time: 11.24 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -90.3400497, 71.5947342, -99.5100861, 78.8561249, -169.1961670, 171.1047974
1: -74.2252808, 62.2856483, -81.7980652, 68.6628799, -142.8881226, 144.0836792
2: -96.2722855, 61.1129951, -106.2715073, 67.5551605, -163.8274384, 167.3844910
3: -107.4888916, 52.9924812, -118.1785278, 58.5795441, -166.0684357, 171.1710052
4: -97.7289276, 74.1748886, -107.5220261, 81.6292343, -179.3581390, 181.6968842
5: -86.0576477, 67.0392990, -94.7736588, 73.8529282, -159.9105530, 161.8129578
6: -81.6614151, 80.3138733, -90.0609818, 88.3588028, -170.0202179, 170.3748474
7: -92.0695267, 70.8811951, -101.1689453, 78.4776840, -170.5471954, 172.0500946
8: -105.4943771, 72.1994629, -116.5465775, 79.8425217, -185.3368988, 188.7460327
9: -81.2101288, 80.6135406, -89.4862442, 88.7901611, -170.0002747, 170.0997620

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3367648, upper bound: 195.3365428
time: 13.28 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3357219, upper bound: 195.3357219
time: 11.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.44 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3205815, upper bound: 195.3179271
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3199324, upper bound: 195.3172984
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3210806, upper bound: 195.3183218
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3204498, upper bound: 195.3176326
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3201766, upper bound: 195.3176522
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3195596, upper bound: 195.3170439
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3206493, upper bound: 195.3180246
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3200305, upper bound: 195.3173514
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3369541, upper bound: 195.3366497
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3359791, upper bound: 195.3358668
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3373022, upper bound: 195.3368271
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3362465, upper bound: 195.3360153
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3363707, upper bound: 195.3363519
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3354469, upper bound: 195.3355856
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3367648, upper bound: 195.3365428
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 26.44
Output dim: 7, lower bound: -195.3357219, upper bound: 195.3357219
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=207.82974243164062
rel_dist={7: [-195.3643469135574, 195.36434691615057]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3542641, upper bound: 195.3531240
time: 13.27 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3599680, upper bound: 195.3599680
time: 13.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 27.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 27.13
Output dim: 7, lower bound: -195.3542641, upper bound: 195.3531240
IS_A2, status: Status.UNKNOWN, split count: 1, time: 27.13
Output dim: 7, lower bound: -195.3599680, upper bound: 195.3599680

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -92.4082718, 73.1420441, -107.7717209, 85.3595963, -177.7678375, 180.9137421
1: -75.9085846, 63.7761002, -88.7133636, 74.5668335, -150.4754181, 152.4894562
2: -98.5751877, 62.5427284, -115.5072174, 73.6027527, -172.1779480, 178.0499420
3: -110.0107346, 54.3026886, -127.8752441, 63.7882919, -173.7990265, 182.1779175
4: -99.9523087, 75.8515472, -116.4223862, 88.5078278, -188.4601288, 192.2739258
5: -87.9972458, 68.5688782, -102.7419586, 80.2015076, -168.1987305, 171.3108063
6: -83.5991364, 82.1437836, -97.7363815, 95.6473083, -179.2464294, 179.8801575
7: -94.1258621, 72.6419296, -109.4475250, 85.7699356, -179.8957825, 182.0894470
8: -107.8815155, 73.8112030, -126.5352020, 86.6466675, -194.5281830, 200.3464050
9: -83.1070404, 82.4340668, -97.1746368, 96.2831192, -179.3901672, 179.6087036

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3495194, upper bound: 195.3478883
time: 13.43 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3488285, upper bound: 195.3474829
time: 14.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -104.9692230, 83.1579132, -112.5857773, 89.2028198, -194.1720276, 195.7436829
1: -86.3696899, 72.5907288, -92.7665253, 77.9590912, -164.3287506, 165.3572083
2: -112.4341278, 71.6278458, -120.8259888, 77.0662308, -189.5003204, 192.4537964
3: -124.5501099, 62.0677834, -133.4589691, 66.7743683, -191.3244629, 195.5267334
4: -113.4068680, 86.2109833, -121.5866852, 92.4893875, -205.8962555, 207.7976532
5: -100.0609283, 78.1082764, -107.3745117, 83.8493423, -183.9102783, 185.4827728
6: -95.1717758, 93.1705627, -102.1655960, 99.8869705, -195.0587463, 195.3361359
7: -106.6246414, 83.4336395, -114.2539062, 89.8811493, -196.5057831, 197.6875458
8: -123.1846008, 84.3752060, -132.3921051, 90.6885376, -213.8731384, 216.7673035
9: -94.6163712, 93.7728882, -101.5780945, 100.6481628, -195.2645264, 195.3509674

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3531240, upper bound: 195.3542641
time: 11.93 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3531240, upper bound: 195.3599680
time: 14.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.64 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.64
Output dim: 7, lower bound: -195.3495194, upper bound: 195.3478883
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.64
Output dim: 7, lower bound: -195.3488285, upper bound: 195.3474829
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.64
Output dim: 7, lower bound: -195.3531240, upper bound: 195.3542641
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.64
Output dim: 7, lower bound: -195.3531240, upper bound: 195.3599680

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -90.7997284, 71.8787079, -101.1258087, 80.1273346, -170.9270477, 173.0045013
1: -74.5683670, 62.6368065, -83.1486893, 69.8598328, -144.4281921, 145.7854614
2: -96.8067703, 61.3959160, -108.1827774, 68.8833237, -165.6900940, 169.5786896
3: -108.1045303, 53.3194885, -120.0123291, 59.7114449, -167.8159790, 173.3318024
4: -98.2124481, 74.5247955, -109.2281189, 82.9963913, -181.2088318, 183.7528992
5: -86.4635239, 67.3470612, -96.3845673, 75.1637573, -161.6272888, 163.7316132
6: -82.1219406, 80.7208557, -91.6331100, 89.7545319, -171.8764496, 172.3539734
7: -92.5059662, 71.2733154, -102.7445602, 80.1360245, -172.6419983, 174.0178833
8: -105.9612885, 72.5031281, -118.5802307, 81.2335739, -187.1948547, 191.0833130
9: -81.6182938, 80.9813385, -91.0427551, 90.2713547, -171.8896484, 172.0240936

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3409789, upper bound: 195.3387891
time: 16.15 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3411790, upper bound: 195.3388653
time: 13.18 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -90.0960846, 71.3236618, -103.0429230, 81.6187439, -171.7148285, 174.3665771
1: -73.9810181, 62.1339645, -84.7032700, 71.1211014, -145.1021118, 146.8371887
2: -96.0226746, 60.8762016, -110.1171265, 69.9856567, -166.0083160, 170.9932861
3: -107.2749176, 52.8763313, -122.3526306, 60.7071991, -167.9821167, 175.2289429
4: -97.4483719, 73.9401855, -111.3046341, 84.5017166, -181.9500885, 185.2448120
5: -85.7846603, 66.8010483, -98.1202011, 76.4612656, -162.2459259, 164.9212494
6: -81.4638062, 80.0969543, -93.2823486, 91.4717331, -172.9355469, 173.3793030
7: -91.7929077, 70.6506042, -104.6904831, 81.3577118, -173.1506042, 175.3410950
8: -105.1135330, 71.9274750, -120.7337265, 82.7026749, -187.8161621, 192.6611938
9: -80.9587631, 80.3380585, -92.6811066, 91.9114838, -172.8702240, 173.0191345

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3404205, upper bound: 195.3383644
time: 14.06 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3405491, upper bound: 195.3384313
time: 12.79 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -104.9692230, 83.1579132, -92.4082718, 73.1420441, -178.1112518, 175.5661774
1: -86.3696899, 72.5907288, -75.9085846, 63.7761002, -150.1457672, 148.4992981
2: -112.4341278, 71.6278458, -98.5751877, 62.5427284, -174.9768372, 170.2030029
3: -124.5501099, 62.0677834, -110.0107346, 54.3026886, -178.8527832, 172.0785217
4: -113.4068680, 86.2109833, -99.9523087, 75.8515472, -189.2584229, 186.1632690
5: -100.0609283, 78.1082764, -87.9972458, 68.5688782, -168.6297913, 166.1054993
6: -95.1717758, 93.1705627, -83.5991364, 82.1437836, -177.3155518, 176.7696533
7: -106.6246414, 83.4336395, -94.1258621, 72.6419296, -179.2665710, 177.5595093
8: -123.1846008, 84.3752060, -107.8815155, 73.8112030, -196.9958038, 192.2567139
9: -94.6163712, 93.7728882, -83.1070404, 82.4340668, -177.0504456, 176.8799286

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3478883, upper bound: 195.3495193
time: 15.42 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3474829, upper bound: 195.3488286
time: 11.91 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -104.9692230, 83.1579132, -104.9692230, 83.1579132, -188.1271362, 188.1271362
1: -86.3696899, 72.5907288, -86.3696899, 72.5907288, -158.9603577, 158.9603577
2: -112.4341278, 71.6278458, -112.4341278, 71.6278458, -184.0619202, 184.0619202
3: -124.5501099, 62.0677834, -124.5501099, 62.0677834, -186.6178741, 186.6178589
4: -113.4068680, 86.2109833, -113.4068680, 86.2109833, -199.6178436, 199.6178436
5: -100.0609283, 78.1082764, -100.0609283, 78.1082764, -178.1691895, 178.1691895
6: -95.1717758, 93.1705627, -95.1717758, 93.1705627, -188.3423157, 188.3423157
7: -106.6246414, 83.4336395, -106.6246414, 83.4336395, -190.0582886, 190.0582886
8: -123.1846008, 84.3752060, -123.1846008, 84.3752060, -207.5598145, 207.5598145
9: -94.6163712, 93.7728882, -94.6163712, 93.7728882, -188.3892517, 188.3892517

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3478883, upper bound: 195.3551878
time: 14.28 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3474829, upper bound: 195.3488285
time: 12.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.31 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 7, lower bound: -195.3409789, upper bound: 195.3387891
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 7, lower bound: -195.3411790, upper bound: 195.3388653
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 7, lower bound: -195.3404205, upper bound: 195.3383644
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 7, lower bound: -195.3405491, upper bound: 195.3384313
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 7, lower bound: -195.3478883, upper bound: 195.3495193
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 7, lower bound: -195.3474829, upper bound: 195.3488286
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 7, lower bound: -195.3478883, upper bound: 195.3551878
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 7, lower bound: -195.3474829, upper bound: 195.3488285

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -76.9347382, 60.9538994, -97.5208740, 77.2781982, -154.2129364, 158.4747467
1: -63.1502419, 52.9757767, -80.1671295, 67.3432922, -130.4935303, 133.1428986
2: -81.6388397, 51.5931511, -104.2259445, 66.3431320, -147.9819641, 155.8190918
3: -91.8945007, 44.8830605, -115.8038330, 57.5134277, -149.4079285, 160.6868896
4: -83.3965149, 63.1803322, -105.3705215, 80.0352097, -163.4317322, 168.5508423
5: -73.2622528, 56.9500389, -92.9389496, 72.4611893, -145.7234497, 149.8889923
6: -69.3793411, 68.5629807, -88.3238678, 86.5864258, -155.9657593, 156.8868408
7: -78.6873474, 59.6045952, -99.1551743, 77.1090851, -155.7964172, 158.7597656
8: -89.3335114, 61.1124229, -114.2443619, 78.2590637, -167.5925598, 175.3567810
9: -68.9692688, 68.6004028, -87.7614517, 87.0463409, -156.0155945, 156.3618469

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3376573, upper bound: 195.3375963
time: 10.32 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3376573, upper bound: 195.3387891
time: 13.22 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -80.0195084, 63.3719368, -96.0790863, 76.1416779, -156.1611786, 159.4510193
1: -65.6793747, 55.0759773, -78.9724731, 66.3366318, -132.0160065, 134.0484467
2: -84.9274979, 53.6089973, -102.6505203, 65.3242111, -150.2517090, 156.2595215
3: -95.5820923, 46.6342697, -114.1130447, 56.6297112, -152.2117920, 160.7473145
4: -86.7236633, 65.7093658, -103.8161926, 78.8606339, -165.5842896, 169.5255432
5: -76.1902695, 59.2041512, -91.5680237, 71.3853455, -147.5755615, 150.7721710
6: -72.1519089, 71.3007126, -87.0062637, 85.3141403, -157.4660492, 158.3069763
7: -81.8256836, 61.9751778, -97.7155380, 75.9123917, -157.7380676, 159.6907196
8: -92.9162369, 63.5171890, -112.5154572, 77.0664902, -169.9827271, 176.0326538
9: -71.7456360, 71.3334579, -86.4500656, 85.7610779, -157.5067139, 157.7835236

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3377761, upper bound: 195.3376724
time: 12.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3377761, upper bound: 195.3376724
time: 11.98 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -76.2640533, 60.4277840, -99.4465332, 78.7755737, -155.0396118, 159.8743134
1: -62.5959244, 52.4995499, -81.7294769, 68.6092377, -131.2051697, 134.2290344
2: -80.8933258, 51.0986671, -106.1693268, 67.4475098, -148.3408356, 157.2679901
3: -91.1064987, 44.4610558, -118.1551514, 58.5136490, -149.6201324, 162.6161804
4: -82.6695251, 62.6237221, -107.4581299, 81.5496521, -164.2191772, 170.0818329
5: -72.6196136, 56.4275818, -94.6828918, 73.7624512, -146.3820648, 151.1104431
6: -68.7536011, 67.9732895, -89.9799957, 88.3113937, -157.0649872, 157.9532776
7: -78.0067291, 59.0100060, -101.1113586, 78.3337784, -156.3405151, 160.1213684
8: -88.5296173, 60.5657883, -116.4109344, 79.7323837, -168.2619934, 176.9767151
9: -68.3388977, 67.9883041, -89.4040070, 88.6930618, -157.0319519, 157.3922882

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3370182, upper bound: 195.3370276
time: 13.92 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3370182, upper bound: 195.3383644
time: 12.52 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -79.3853149, 62.8721542, -98.0431900, 77.6693420, -157.0546112, 160.9153442
1: -65.1529083, 54.6247673, -80.5660095, 67.6301346, -132.7830200, 135.1907806
2: -84.2203827, 53.1393166, -104.6362686, 66.4555817, -150.6759491, 157.7755890
3: -94.8356323, 46.2350426, -116.5094833, 57.6530075, -152.4886322, 162.7445221
4: -86.0368271, 65.1816635, -105.9463501, 80.4056091, -166.4424438, 171.1279907
5: -75.5786133, 58.7112999, -93.3476181, 72.7153091, -148.2939148, 152.0589142
6: -71.5579987, 70.7407913, -88.6991272, 87.0741730, -158.6321411, 159.4399109
7: -81.1825790, 61.4111824, -99.7113953, 77.1678467, -158.3504181, 161.1225739
8: -92.1538010, 63.0004082, -114.7283554, 78.5707855, -170.7245789, 177.7287598
9: -71.1508484, 70.7532654, -88.1286087, 87.4421921, -158.5930328, 158.8818512

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3371673, upper bound: 195.3371673
time: 10.93 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3371673, upper bound: 195.3384313
time: 12.06 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -98.2581329, 77.8778534, -90.7997284, 71.8787079, -170.1368408, 168.6775818
1: -80.7665787, 67.8431778, -74.5683670, 62.6368065, -143.4033813, 142.4115448
2: -105.0398788, 66.8529282, -96.8067703, 61.3959160, -166.4357910, 163.6596985
3: -116.6141815, 57.9503212, -108.1045303, 53.3194885, -169.9336548, 166.0548248
4: -106.1369934, 80.6508102, -98.2124481, 74.5247955, -180.6617737, 178.8632507
5: -93.6411743, 73.0181885, -86.4635239, 67.3470612, -160.9882355, 159.4816589
6: -89.0047150, 87.2230759, -82.1219406, 80.7208557, -169.7255707, 169.3450012
7: -99.8584290, 77.7346115, -92.5059662, 71.2733154, -171.1317291, 170.2405701
8: -115.1531677, 78.9095306, -105.9612885, 72.5031281, -187.6562500, 184.8708191
9: -88.4194183, 87.7053452, -81.6182938, 80.9813385, -169.4007568, 169.3236389

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3387891, upper bound: 195.3409789
time: 14.88 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3388653, upper bound: 195.3411790
time: 12.48 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -100.3584137, 79.5168610, -90.0960846, 71.3236618, -171.6820526, 169.6129303
1: -82.4760895, 69.2385178, -73.9810181, 62.1339645, -144.6100159, 143.2195435
2: -107.1818695, 68.0916748, -96.0226746, 60.8762016, -168.0580444, 164.1143341
3: -119.1716614, 59.0648499, -107.2749176, 52.8763313, -172.0479889, 166.3397675
4: -108.4165573, 82.3142853, -97.4483719, 73.9401855, -182.3567505, 179.7626648
5: -95.5569153, 74.4586639, -85.7846603, 66.8010483, -162.3579559, 160.2433167
6: -90.8289948, 89.1049042, -81.4638062, 80.0969543, -170.9259338, 170.5687103
7: -101.9972534, 79.1226959, -91.7929077, 70.6506042, -172.6478577, 170.9155731
8: -117.5351562, 80.5285339, -105.1135330, 71.9274750, -189.4626160, 185.6420593
9: -90.2338104, 89.5166550, -80.9587631, 80.3380585, -170.5718689, 170.4754028

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 153

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3383644, upper bound: 195.3404205
time: 14.62 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3384313, upper bound: 195.3405491
time: 13.20 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -98.2581329, 77.8778534, -103.2620392, 81.8142395, -180.0723724, 181.1398926
1: -80.7665787, 67.8431778, -84.9429169, 71.3817062, -152.1482849, 152.7861023
2: -105.0398788, 66.8529282, -110.5526733, 70.4149323, -175.4547729, 177.4056091
3: -116.6141815, 57.9503212, -122.5298462, 61.0203476, -177.6345062, 180.4801331
4: -106.1369934, 80.6508102, -111.5582047, 84.7949371, -190.9319153, 192.2090149
5: -93.6411743, 73.0181885, -98.4284363, 76.8135147, -170.4546814, 171.4465637
6: -89.0047150, 87.2230759, -93.6030273, 91.6569519, -180.6616364, 180.8261108
7: -99.8584290, 77.7346115, -104.9021835, 81.9856873, -181.8441162, 182.6367950
8: -115.1531677, 78.9095306, -121.1404724, 82.9840698, -198.1372223, 200.0499878
9: -88.4194183, 87.7053452, -93.0404282, 92.2291718, -180.6485748, 180.7457733

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3470615, upper bound: 195.3473220
time: 11.34 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3469832, upper bound: 195.3473348
time: 12.13 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -100.3584137, 79.5168610, -102.6177979, 81.3055954, -181.6640015, 182.1346283
1: -82.4760895, 69.2385178, -84.4033585, 70.9211807, -153.3972778, 153.6418610
2: -107.1818695, 68.0916748, -109.8328857, 69.9443207, -177.1261902, 177.9245300
3: -119.1716614, 59.0648499, -121.7705994, 60.6146317, -179.7862701, 180.8354492
4: -108.4165573, 82.3142853, -110.8585358, 84.2569427, -192.6734924, 193.1728210
5: -95.5569153, 74.4586639, -97.8050842, 76.3160934, -171.8730011, 172.2637482
6: -90.8289948, 89.1049042, -93.0026855, 91.0840454, -181.9129944, 182.1075897
7: -101.9972534, 79.1226959, -104.2499237, 81.4210434, -183.4182892, 183.3725891
8: -117.5351562, 80.5285339, -120.3624344, 82.4573212, -199.9924774, 200.8909607
9: -90.2338104, 89.5166550, -92.4398193, 91.6416016, -181.8754120, 181.9564667

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3467329, upper bound: 195.3465962
time: 11.00 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -195.3465866, upper bound: 195.3465869
time: 13.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.61 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3376573, upper bound: 195.3375963
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3376573, upper bound: 195.3387891
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3377761, upper bound: 195.3376724
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3377761, upper bound: 195.3376724
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3370182, upper bound: 195.3370276
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3370182, upper bound: 195.3383644
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3371673, upper bound: 195.3371673
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3371673, upper bound: 195.3384313
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3387891, upper bound: 195.3409789
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3388653, upper bound: 195.3411790
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3383644, upper bound: 195.3404205
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3384313, upper bound: 195.3405491
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3470615, upper bound: 195.3473220
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3469832, upper bound: 195.3473348
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3467329, upper bound: 195.3465962
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.61
Output dim: 7, lower bound: -195.3465866, upper bound: 195.3465869

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -94.6740723, 75.0449905, -76.9347382, 60.9538994, -155.6279755, 151.9797058
1: -77.8029175, 65.3411942, -63.1502419, 52.9757767, -130.7786865, 128.4914398
2: -101.1094894, 64.3244476, -81.6388397, 51.5931511, -152.7026367, 145.9632721
3: -112.4291077, 55.7658997, -91.8945007, 44.8830605, -157.3121643, 147.6603851
4: -102.3038864, 77.7117767, -83.3965149, 63.1803322, -165.4842224, 161.1082916
5: -90.2154388, 70.3295135, -73.2622528, 56.9500389, -147.1654663, 143.5917511
6: -85.7139893, 84.0754166, -69.3793411, 68.5629807, -154.2769470, 153.4547424
7: -96.2926254, 74.7235413, -78.6873474, 59.6045952, -155.8972168, 153.4108734
8: -110.8507690, 75.9502640, -89.3335114, 61.1124229, -171.9631805, 165.2837830
9: -85.1554337, 84.4998398, -68.9692688, 68.6004028, -153.7558289, 153.4690857

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3190734, upper bound: 195.3222858
time: 13.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3181950, upper bound: 195.3213890
time: 12.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -93.2211990, 73.9012833, -80.0195084, 63.3719368, -156.5931396, 153.9207916
1: -76.6007385, 64.3249817, -65.6793747, 55.0759773, -131.6767120, 130.0043488
2: -99.5210114, 63.2955093, -84.9274979, 53.6089973, -153.1300049, 148.2230072
3: -110.7256241, 54.8730850, -95.5820923, 46.6342697, -157.3598938, 150.4551544
4: -100.7351379, 76.5268631, -86.7236633, 65.7093658, -166.4445038, 163.2505188
5: -88.8349838, 69.2454605, -76.1902695, 59.2041512, -148.0391388, 145.4356995
6: -84.3829575, 82.7943878, -72.1519089, 71.3007126, -155.6836700, 154.9462891
7: -94.8392181, 73.5153809, -81.8256836, 61.9751778, -156.8143921, 155.3410339
8: -109.1059647, 74.7481461, -92.9162369, 63.5171890, -172.6231537, 167.6643829
9: -83.8324890, 83.2005310, -71.7456360, 71.3334579, -155.1659546, 154.9461670

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3193920, upper bound: 195.3227234
time: 14.02 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3184593, upper bound: 195.3218369
time: 15.01 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -96.7790298, 76.6873779, -76.2640533, 60.4277840, -157.2067871, 152.9514313
1: -79.5168610, 66.7386856, -62.5959244, 52.4995499, -132.0163879, 129.3346100
2: -103.2551041, 65.5646515, -80.8933258, 51.0986671, -154.3537750, 146.4579773
3: -114.9923553, 56.8819389, -91.1064987, 44.4610558, -159.4534149, 147.9884338
4: -104.5902863, 79.3790741, -82.6695251, 62.6237221, -167.2140045, 162.0485992
5: -92.1360626, 71.7717514, -72.6196136, 56.4275818, -148.5636292, 144.3913574
6: -87.5416946, 85.9614334, -68.7536011, 67.9732895, -155.5149689, 154.7150269
7: -98.4358444, 76.1114197, -78.0067291, 59.0100060, -157.4458466, 154.1181335
8: -113.2368927, 77.5721588, -88.5296173, 60.5657883, -173.8026733, 166.1017761
9: -86.9713821, 86.3144226, -68.3388977, 67.9883041, -154.9596558, 154.6533203

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3187081, upper bound: 195.3217384
time: 15.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3178612, upper bound: 195.3208957
time: 13.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -95.3593521, 75.5693970, -79.3853149, 62.8721542, -158.2315063, 154.9546814
1: -78.3421021, 65.7465286, -65.1529083, 54.6247673, -132.9668732, 130.8993988
2: -101.7035294, 64.5586319, -84.2203827, 53.1393166, -154.8428345, 148.7790222
3: -113.3275528, 56.0094299, -94.8356323, 46.2350426, -159.5625916, 150.8450623
4: -103.0579147, 78.2208862, -86.0368271, 65.1816635, -168.2395477, 164.2577209
5: -90.7869492, 70.7111053, -75.5786133, 58.7112999, -149.4982452, 146.2897034
6: -86.2419968, 84.7101746, -71.5579987, 70.7407913, -156.9827576, 156.2681580
7: -97.0168915, 74.9302750, -81.1825790, 61.4111824, -158.4280701, 156.1128540
8: -111.5312195, 76.3963242, -92.1538010, 63.0004082, -174.5316315, 168.5501251
9: -85.6799240, 85.0450897, -71.1508484, 70.7532654, -156.4331665, 156.1959381

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3189928, upper bound: 195.3221505
time: 12.24 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3180793, upper bound: 195.3213024
time: 16.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -94.6740723, 75.0449905, -88.8552704, 70.4370422, -165.1111145, 163.9002380
1: -77.8029175, 65.3411942, -73.0332642, 61.3271446, -139.1300659, 138.3744354
2: -101.1094894, 64.3244476, -94.7607040, 60.2575951, -161.3670807, 159.0851440
3: -112.4291077, 55.7658997, -105.6952133, 52.2337494, -164.6628571, 161.4611206
4: -102.3038864, 77.7117767, -96.1452179, 72.9786453, -175.2825317, 173.8569946
5: -90.2154388, 70.3295135, -84.6664047, 66.0108032, -156.2262268, 154.9958801
6: -85.7139893, 84.0754166, -80.3685760, 78.9998779, -164.7138367, 164.4439697
7: -96.2926254, 74.7235413, -90.5552063, 69.8966446, -166.1892700, 165.2787476
8: -110.8507690, 75.9502640, -103.8313293, 71.1049576, -181.9556732, 179.7815704
9: -85.1554337, 84.4998398, -79.9256668, 79.3430252, -164.4984436, 164.4255066

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3370503, upper bound: 195.3374496
time: 12.12 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3360977, upper bound: 195.3362268
time: 11.19 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -93.2211990, 73.9012833, -91.9157181, 72.8341293, -166.0553284, 165.8170013
1: -76.6007385, 64.3249817, -75.5394745, 63.4082069, -140.0089417, 139.8644257
2: -99.5210114, 63.2955093, -98.0195007, 62.2617035, -161.7827148, 161.3150024
3: -110.7256241, 54.8730850, -109.3496933, 53.9750481, -164.7006531, 164.2227325
4: -100.7351379, 76.5268631, -99.4371338, 75.4809189, -176.2160492, 175.9639893
5: -88.8349838, 69.2454605, -87.5708237, 68.2527466, -157.0877380, 156.8162689
6: -84.3829575, 82.7943878, -83.1232758, 81.7098389, -166.0927734, 165.9176636
7: -94.8392181, 73.5153809, -93.6602020, 72.2560654, -167.0952606, 167.1755524
8: -109.1059647, 74.7481461, -107.3870850, 73.4857483, -182.5917053, 182.1352234
9: -83.8324890, 83.2005310, -82.6798553, 82.0477448, -165.8802338, 165.8803864

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3371178, upper bound: 195.3376827
time: 15.14 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3361409, upper bound: 195.3364108
time: 11.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -96.7790298, 76.6873779, -88.2296906, 69.9447784, -166.7238007, 164.9170685
1: -79.5168610, 66.7386856, -72.5116577, 60.8796921, -140.3965454, 139.2503357
2: -103.2551041, 65.5646515, -94.0646896, 59.7976265, -163.0527344, 159.6293182
3: -114.9923553, 56.8819389, -104.9588699, 51.8405800, -166.8329315, 161.8408051
4: -104.5902863, 79.3790741, -95.4666290, 72.4595337, -177.0498199, 174.8457031
5: -92.1360626, 71.7717514, -84.0637512, 65.5262070, -157.6622467, 155.8355103
6: -87.5416946, 85.9614334, -79.7852249, 78.4452972, -165.9869995, 165.7466583
7: -98.4358444, 76.1114197, -89.9228592, 69.3452606, -167.7810974, 166.0342712
8: -113.2368927, 77.5721588, -103.0788422, 70.5944748, -183.8313599, 180.6509857
9: -86.9713821, 86.3144226, -79.3400955, 78.7726822, -165.7440491, 165.6545105

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3366684, upper bound: 195.3367758
time: 13.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -195.3357362, upper bound: 195.3355806
time: 11.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.60 seconds
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3190734, upper bound: 195.3222858
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3181950, upper bound: 195.3213890
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3193920, upper bound: 195.3227234
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3184593, upper bound: 195.3218369
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3187081, upper bound: 195.3217384
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3178612, upper bound: 195.3208957
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3189928, upper bound: 195.3221505
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3180793, upper bound: 195.3213024
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3370503, upper bound: 195.3374496
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3360977, upper bound: 195.3362268
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3371178, upper bound: 195.3376827
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3361409, upper bound: 195.3364108
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3366684, upper bound: 195.3367758
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.60
Output dim: 7, lower bound: -195.3357362, upper bound: 195.3355806
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.60
Output dim: 7, lower bound: -195.3465866, upper bound: 195.3465869
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=207.82974243164062
rel_dist={7: [-195.36437973784498, 195.3643797290814]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1677.13 seconds
